"""Compile evidence-first artifacts into a portable draft author Skill.

The current input contracts are work-scoped, so this compiler must publish an
honest ``draft`` package.  Cross-work author traits and release evidence are
represented explicitly rather than inferred from one work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from .contracts import AuthorSkillBundle, AuthorStyleProfile, ChapterAnnotation, NarrativeTemplateLibrary, SkillQualification, WorkContinuity
from .jsonio import read_jsonl, write_json
from .paths import corpus_dir, output_dir
from .skill_package import (
    assess_skill_directory,
    build_draft_references,
    build_openai_yaml,
    build_skill_markdown,
    normalize_skill_name,
    select_style_context_script,
    self_check_script,
)
from .style_distillation import assess_style_profile
from .style_execution import style_execution_spec
from .template_mining import assess_template_library


DEFAULT_SKILL_LIMIT = 20
_CORE_FAMILIES = frozenset({"sentence_rhythm", "paragraph_pacing", "dialogue", "transition", "focalization"})
_LEGACY_OUTPUT_FILES = (
    "author_narrative_profile.json", "chapter_state_schema.json", "distillation_batches.jsonl",
    "skill_context.json", "template_library.json", "template_mining_batches.jsonl",
    "author_style_profile.json", "narrative_templates.json", "continuity_requirements.json",
    "style_execution_spec.json", "prompt_program.json", "skill_manifest.json", "skill_quality.json",
)


@dataclass(frozen=True)
class SkillPackageIssue:
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SkillPackageQualityReport:
    author_id: str
    work_id: str
    chapter_count: int
    template_count: int
    style_constraint_count: int
    generated_files: tuple[str, ...]
    release_status: str = "draft"
    qualification: SkillQualification = field(default_factory=lambda: SkillQualification(source_work_count=1))
    issues: tuple[SkillPackageIssue, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not any(item.severity == "error" for item in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "2.0",
            "author_id": self.author_id,
            "work_id": self.work_id,
            "chapter_count": self.chapter_count,
            "template_count": self.template_count,
            "style_constraint_count": self.style_constraint_count,
            "generated_files": list(self.generated_files),
            "passed": self.passed,
            "pass_scope": "package_structure_only",
            "release_status": self.release_status,
            "validated_for_release": self.passed and self.release_status == "validated",
            "qualification": self.qualification.to_dict(),
            "issues": [item.to_dict() for item in self.issues],
        }


def _artifact_path(folder: Path, stem: str, limit: int, suffix: str) -> Path:
    exact = folder / f"{stem}.sample-{limit}{suffix}"
    if exact.exists():
        return exact
    candidates = sorted(
        path for path in folder.glob(f"{stem}.sample-*{suffix}")
        if ".quality." not in path.name
    )
    if not candidates:
        raise FileNotFoundError(f"required stage artifact is missing: {stem}")
    return candidates[-1]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path.name}")
    return payload


def _source_terms(folder: Path, limit: int) -> tuple[str, ...]:
    path = _artifact_path(folder, "chapter_annotations", limit, ".jsonl")
    result: set[str] = set()
    for row in read_jsonl(path):
        annotation = ChapterAnnotation.from_dict(row)
        for entity in annotation.entities:
            for term in (entity.canonical_name, *entity.aliases):
                value = str(term).strip()
                # Two-character Chinese strings are often generic role labels.
                if len(value) >= 3:
                    result.add(value)
    return tuple(sorted(result, key=len, reverse=True))


def _template_texts(library: NarrativeTemplateLibrary) -> Iterable[str]:
    for item in library.templates:
        yield item.purpose
        yield from item.role_slots
        yield from item.beat_sequence
        yield from item.state_effects
        yield from item.selection_tags
        yield from item.variation_axes


def _continuity_requirements(continuity: WorkContinuity) -> dict[str, Any]:
    """Expose constraints and counts, never the source work's entity/state values."""

    return {
        "schema_version": "1.0",
        "chapter_count": len(continuity.chapter_ids),
        "observed_graph_counts": {
            "global_entity_count": len(continuity.global_entities),
            "timeline_event_count": len(continuity.timeline_events),
            "state_ledger_entry_count": len(continuity.state_ledger),
            "location_transition_count": len(continuity.location_transitions),
            "unlinked_spatial_relation_count": len(continuity.unlinked_spatial_relation_ids),
        },
        "required_original_inputs": [
            "story_state.entities", "story_state.relationships", "story_state.resources",
            "story_state.knowledge", "story_state.time", "story_state.location", "chapter_brief",
        ],
        "hard_checks": [
            "人物身份、关系、资源、知识、时间和位置只能从当前原创故事状态读取。",
            "每一场景先满足进入状态，再写行动、转折和退出状态；不得让后果先于原因出现。",
            "发生移动、信息传播、关系变化或资源消耗时，必须在章节计划的状态变化中显式记录。",
            "没有当前原创输入支持的事实必须保留为未知，不得借用训练样本中的人名、地点、术语或情节补全。",
        ],
    }


def _plan_prompt() -> str:
    example = {
        "chapter_goal": "让原创主角在受限条件下取得局部进展，并留下新的压力。",
        "selected_template_ids": ["library:chapter-001", "library:event-002"],
        "scenes": [{
            "scene_no": 1,
            "entry_state": ["当前原创状态中的已知条件"],
            "objective": "本场景要推进的目标",
            "pressure": "阻力或代价",
            "turn": "让计划改变的转折",
            "exit_state": ["可追踪的新状态"],
        }],
        "chapter_state_changes": [{"slot": "原创状态字段", "before": "已知值", "after": "变化后的值", "cause_scene_no": 1}],
        "covered_required_event_ids": ["target-event-001"],
        "covered_required_state_change_ids": ["target-change-001"],
        "chapter_end_contract": ["在本章目标状态后收束，不提前推进后续章节"],
        "style_budget": {"metrics": [{"metric_id": "mean_sentence_chars", "target": 42, "preferred_min": 36, "preferred_max": 48, "priority": "high"}]},
        "continuity_checks": ["时间、位置、资源和知识均与当前原创状态连续"],
    }
    return f"""# 章节规划提示词

你是中文小说的章节规划器。严格遵循 `<ExecutionMode>` 确定事实边界：默认模式只使用当前原创输入；结构保真评测模式只可使用评测输入明确给出的结构要素。作者 skill 始终只提供抽象结构与文风约束，绝不补入未提供的来源事实或原文句子。

先从 `<NarrativeTemplateLibrary>` 选择一至三个适配的 `template_id`，再把它们落实为可检查的场景状态变化。`<ContinuityRequirements>` 是硬约束；所有状态变化都必须有场景原因。

只输出一个合法 JSON object，顶层字段必须且只能是：`chapter_goal`、`selected_template_ids`、`scenes`、`chapter_state_changes`、`covered_required_event_ids`、`covered_required_state_change_ids`、`chapter_end_contract`、`style_budget`、`continuity_checks`。`style_budget.metrics` 只能选用 `<StyleExecutionSpec>` 中的 `metric_id`，目标必须落在对应的柔性区间内。若 `<ChapterBrief>` 给出了带 `contract_id` 的 `required_events` 或 `required_state_changes`，相应的 `covered_required_*_ids` 必须逐一列全；`chapter_end_contract` 必须说明本章应在哪里收束、不得提前进入哪些未提供事件。没有这类输入时两个数组填空数组。不要输出 Markdown 或解释。

JSON 示例（只说明格式，不可复用其中的人物或事件）：
{json.dumps(example, ensure_ascii=False, indent=2)}

<AuthorStyleProfile>
{{AUTHOR_STYLE_PROFILE_JSON}}
</AuthorStyleProfile>
<NarrativeTemplateLibrary>
{{NARRATIVE_TEMPLATE_LIBRARY_JSON}}
</NarrativeTemplateLibrary>
<ContinuityRequirements>
{{CONTINUITY_REQUIREMENTS_JSON}}
</ContinuityRequirements>
<StyleExecutionSpec>
{{STYLE_EXECUTION_SPEC_JSON}}
</StyleExecutionSpec>
<ExecutionMode>
{{EXECUTION_MODE_INSTRUCTIONS}}
</ExecutionMode>
<OriginalStoryState>
{{ORIGINAL_STORY_STATE_JSON}}
</OriginalStoryState>
<ChapterBrief>
{{CHAPTER_BRIEF_JSON}}
</ChapterBrief>
"""


def _draft_prompt() -> str:
    return """# 原创章节正文提示词

你是中文小说写作者。根据 `<ChapterPlan>` 和当前故事状态写连续正文。严格遵循 `<ExecutionMode>` 确定事实边界：默认模式只能使用当前原创输入；结构保真评测模式只能使用评测输入明确给出的结构要素。任何模式都不得输入、复用或改写原文句子，也不得补入输入中没有的专名、设定或具体情节。

执行 `<AuthorStyleProfile>` 的中文约束时，把 `baselines` 视为柔性观察区间。`<ChapterPlan>.style_budget` 给出了本章优先指标，`<ChapterPlan>.style_execution_directives` 则把这些指标转换为本章必须优先落实的写作动作；`<ChapterPlan>.style_execution_blueprint` 会把篇幅、段落、句子、对话、感知和转场转换为可执行的本章锚点。先在心中按动作与蓝图安排正文，再写正文。若蓝图存在 `chapter_char_count.minimum`，必须完整写到该下限；扩写只可补足计划内场景的行动、阻力、对话、感知或结果。结合 `<StyleExecutionSpec>` 主动调节句段、对话和感知密度，但不得为了凑指标破坏可读性、事件顺序或状态连续性。执行 `<SelectedTemplates>` 时，只复用抽象功能、节拍和状态效果，具体内容必须由本次计划决定。每个场景至少体现一个可验证的行动、阻力、选择或后果，并与计划中的状态变化一致。若计划包含 `covered_required_event_ids`、`covered_required_state_change_ids` 与 `chapter_end_contract`，它们是正文硬约束：必须全部兑现，且不得把后续未覆盖事件写进本章。

只输出正文，不输出标题、解释、Markdown、JSON 或分析。

<AuthorStyleProfile>
{{AUTHOR_STYLE_PROFILE_JSON}}
</AuthorStyleProfile>
<SelectedTemplates>
{{SELECTED_TEMPLATES_JSON}}
</SelectedTemplates>
<ContinuityRequirements>
{{CONTINUITY_REQUIREMENTS_JSON}}
</ContinuityRequirements>
<StyleExecutionSpec>
{{STYLE_EXECUTION_SPEC_JSON}}
</StyleExecutionSpec>
<ExecutionMode>
{{EXECUTION_MODE_INSTRUCTIONS}}
</ExecutionMode>
<OriginalStoryState>
{{ORIGINAL_STORY_STATE_JSON}}
</OriginalStoryState>
<ChapterPlan>
{{CHAPTER_PLAN_JSON}}
</ChapterPlan>
"""


_DRAFT_LENGTH_CONTRACT = """\n\n## 最后长度提交规则\n\n现在只检查 `<本段局部正文门槛>`：`最低正文字符` 是不可降低的硬下限，`目标正文字符` 是应主动靠近的正常篇幅，不是上限。交付 JSON 前请在内部按正文可见字符计数。非短段正文必须不少于最低正文字符，并在不越过剧情边界的前提下尽量接近目标正文字符；未达到下限时不得输出半段或用一句总结收束，必须在本段已有节拍中补足动作、感知、压力和反应的连续过程。补写只能使用段落写作包已经许可的事实、事件和人物。"""


_REPAIR_LENGTH_CONTRACT = """\n\n## 最后长度修复规则\n\n若段落校验结果写有“至少需要 N 字”的局部正文门槛问题，必须输出完整替换段，而非在旧段后机械附一句。请先在内部计数：正文可见字符必须不少于 N，并尽量靠近 `<本段局部正文门槛>.目标正文字符`；没有达到 N 就不得交付。只能在当前已有节拍中延展动作前的注意对象、动作中的身体或环境反馈、压力如何改变判断、反应如何落到已许可的下一步。不得重复同一事实、堆砌形容词，或写入任何后续事件。"""


def _scene_draft_prompt() -> str:
    example = {
        "场景ID": "scene-001",
        "段落正文": [
            {"段落ID": "scene-001:paragraph-00", "正文": "这里是完整的中文小说段落。", "自检兑现事实ID": ["fact-001"], "自检兑现事件ID": []},
            {"段落ID": "scene-001:paragraph-01", "正文": "这里推进动作并呈现结果。", "自检兑现事实ID": ["fact-002"], "自检兑现事件ID": ["event-001"]},
        ],
    }
    return f"""# 场景—段落正文生成提示词

你是中文小说作者。每次请求只生成 `<段落写作包>` 中的一个当前段落；不得书写其他场景或后续段落，不得使用未提供的专名、设定、人物或情节，不得复制、改写或补全任何来源正文句子。写作包中的 `<叙事事实子图>` 是唯一可读取的跨段事实来源：它只包含当前段落获准使用的结构化节点、关系、因果、已验证伏笔状态和已提交状态；禁止索取、输入、引用或依据整章原文。`章节合同窗口` 只开放本段可推进的事件；`本章段落状态簿` 和 `前文交接` 只来自已经通过校验的生成正文。它们用于承接状态和语气，不得改写已成立结果，更不得借此提前写未来事件。必须逐项满足 `<本段局部正文门槛>` 的字符数、短段上限和直接对白要求；这些是本段提交前会立即校验的硬条件。

事实边界不等于禁止小说动作。当前场景的许可人物可以有不改变持久状态的停顿、目光、手势、呼吸、坐立、短暂的感知和环境氛围；这些只用于演出当前节拍，不应被写成新事实。禁止的是会留下新故事账目的断言：新增或改写人物关系/身份/动机，人物跨地点移动或精确位置变化，资源得失，设定与规则，获得关键信息，时间跳转，因果结果，或未来事件已经发生。若当前段没有可推进事件，就用上述低风险动作、感知和已许可事实完成场景定位或承接，不得预告下段事件。

若 `<当前段落程序>.段落功能` 是“场景定位”，只能书写已经成立的场内静态姿态、视线、呼吸、触感或环境；不得提前书写进入、返回、离开、跨越门槛、赶路、收拾或交付等动作。它们只有在当前段明确绑定相应事件 ID 时才可出现。

事实不是可以直接贴上的心理标签。若必须表达的是愿望、情绪、判断或知情，应先用当前已许可人物的可见反应、注意对象或直接对白把它演出来，再让读者得出该事实；不得只写“他很向往”“他心里想着”来代替过程。当前事件的动作含有多个动词时，逐个落实其可见过程，而不是只复述其中一个结果。

这里要求的是小说正文，不是剧情梗概。`戏剧节拍` 是每段必须被演出的最小过程：先让读者看见行动或局面变化，再写人物受到的压力、感知或回应，最后才让状态变化成立。不得把关键节拍压缩成“甲得知消息后决定离开”“甲说出消息，于是众人同意”这类结果句。下面是匿名的写法对照：

不合格：甲得知消息后，决定离开。

合格：甲先从对方的停顿、动作或一句对白中确认消息；消息造成的压力使他的注意力或身体反应发生变化；他再以可见动作、对白或取舍落实离开的选择。

每个段落程序绑定的 `节拍ID列表` 都必须在该段中落实其 `可见动作`、`视角细节`、`压力或阻力` 和 `状态变化`。没有压力的节拍也必须让动作造成局面变化；有 `对白压力` 的节拍必须写出施压与回应，且回应影响下一步行动。感知和环境细节必须改变人物的注意力、判断或动作，不能只堆砌形容词。一个段落只展开自己的节拍和一至两个紧密事实，不能把后续结果提前概述。

`叙事任务` 不是供参考的摘要，而是本段必须完成的深层作用：人物欲望要在限制下产生可见选择，因果压力要改变后续行动，信息控制要让读者只在正确时点知道正确信息，对照要通过可比较的动作或处境成立，情绪转折要有触发和反应。`反事实守卫` 描述删掉这一安排会损失什么；正文完成后若仍可整段删除而不发生该项损失，就说明只复述了事实、没有完成机制。禁止直接把“叙事任务”原句写进正文，必须用当前许可的事实、事件、动作、感知和对白实现它。

当前写作包只对应一个段落，因此 `段落正文` 数组必须只含一项，并保留原有 `段落ID`。正文要落实该段的“段落功能”、必须表达的事实 ID 和事件 ID；事实可以通过叙述、行动、对话或心理呈现，但不得改变谓词方向、人物关系、状态结果或因果顺序。写作包未许可的后续事件不得写成已发生、完整预告或回顾为既成事实。

若 `<段落写作包>` 的 `生成模式` 为“受控扩写”，必须同时执行每个戏剧节拍的 `扩写许可` 和 `依据事实ID`：

- “原文复现”只落实已经列出的事实或事件；
- “合理推导”只能补动作准备、感知、反应、对白承接或已知结果的余波，不能新增人物关系、具体动机、资源、地点、时间、设定或事件结果；
- “氛围扩写”只能补当前人物已能感知的环境、距离、停顿或节奏，不能产生新的剧情信息。

章节目标字数由程序拆入各段的 `目标字数` 与 `最低字数`；不得通过重复事实、空泛心理、无效环境描写或把未来事件提前写出凑字数。每一段只扩展其自身绑定的节拍，且必须能指出其依据事实 ID。

按 `<风格执行蓝图>` 调整节奏、段长、对话和感知密度；其中若给出 `段落字符范围` 与 `句子字符范围`，每段应主动靠近该范围，优先用一到两个完整句承载因果，避免连续碎短句或无意义扩写。若给出 `段落写作预算`，必须按每个 `段落ID` 的目标字数、最大字数、建议句数、短段标记、`职责标签` 和 `叙事职责`执行；其中职责是逐段可检测的执行契约，缺失就视为本段未完成。被分配“对话”职责的段落必须以中文引号“”写出不少于 `最低对白字数` 的直接对白；被分配“转场”职责的段落必须出现“随后、此时、片刻后、不久、旋即、继而、当即”之一；被分配“感知”职责的段落必须出现“看到、看见、望着、听到、感到、察觉”之一。对话只能使用当前场景已列出的参与者，转场和感知只能承接当前程序已有的状态、行动与事实；不得为满足职责引入新人物、新事实或新事件。短段只在被标记的段落自然收束，不能拆散事实链。它们仍是柔性范围，事实、因果与可读性优先。正文只写当前场景，不写标题、解释、分析或 Markdown。

只输出一个合法 JSON object，顶层字段必须且只能是 `场景ID`、`段落正文`。`段落正文` 必须是数组；每项字段必须且只能是 `段落ID`、`正文`、`自检兑现事实ID`、`自检兑现事件ID`。两个自检数组必须列出本段实际兑现的 ID，不得编造 ID。`正文` 必须是单行 JSON 字符串；小说对白只能使用中文引号“”，不得在正文中使用未转义的 ASCII 双引号。禁止输出空对象 `{{}}`、空数组或省略任一顶层字段；若内容很短，也必须完整填写下面示例中的字段结构。JSON 示例：
{json.dumps(example, ensure_ascii=False, indent=2)}

最后执行规则：`转场` 与 `感知` 职责是语义职责，不要求或鼓励机械出现某个承接词、感知词；必须通过前一状态到当前压力的因果衔接，以及会影响判断的具体感知来实现。只有 `对白` 与段长仍由程序进行可见格式检测。若前文的词汇示例与此规则冲突，以本条为准。

<作者风格档案>
{{AUTHOR_STYLE_PROFILE_JSON}}
</作者风格档案>
<已选抽象模板>
{{SELECTED_TEMPLATES_JSON}}
</已选抽象模板>
<风格执行蓝图>
{{STYLE_EXECUTION_BLUEPRINT_JSON}}
</风格执行蓝图>
<本段局部正文门槛>
{{PARAGRAPH_LOCAL_REQUIREMENTS_JSON}}
</本段局部正文门槛>
<段落写作包>
{{PARAGRAPH_WRITER_PACKET_JSON}}
</段落写作包>
""" + _DRAFT_LENGTH_CONTRACT


def _paragraph_validate_prompt() -> str:
    example = {
        "段落ID": "scene-001:paragraph-01",
        "通过": False,
        "已实现事实ID": ["fact-002"],
        "缺失事实ID": [],
        "已实现事件ID": [],
        "缺失事件ID": ["event-001"],
        "已实现叙事机制ID": [],
        "缺失叙事机制ID": ["mechanism-001"],
        "未授权断言": ["把未开放的后续冲突写成已经发生"],
        "问题": ["事件只被概述为结果，缺少当前段可见行动"],
    }
    return f"""# 单段落语义校验提示词

你是中文小说单段校验器。只比较 `<段落写作包>` 与 `<待校验段落>`；它们不包含也不允许你索取任何原文章节、完整场景正文或未来段落。不得相信正文中的“自检兑现”数组，必须仅根据正文逐项核验当前段是否真正兑现了分配给它的事实和事件，是否把未开放事件、关系、结果、时间或空间变化写成既成事实。还要核验“当前段落节拍”是否被演成行动、压力、反应或选择过程；若只用“得知、决定、于是、最终”等结果句概述节拍，必须写入“问题”。逐项核验每个节拍的 `叙事任务`：正文必须通过具体安排产生指定作用，不能仅出现相关事实或把任务换句话复述。使用 `反事实守卫` 做删除测试；若删掉承担该机制的句子后，人物选择、因果压力、信息时点、对照或情绪转折并无损失，必须把对应机制 ID 记为缺失。`章节合同窗口` 中的未来锁定事件绝不可在本段透露、预演或兑现；`本章段落状态簿` 只可承接，不能被改写。

“未授权断言”只登记会改变故事账目的内容：新增/冲突的身份、关系、动机、资源、规则、知识、地点转移、明确时间跳转、因果结果或未来事件。不要把已许可场景人物的短暂姿态、目光、呼吸、坐立、普通感知、无名环境氛围，或不改变地点的同场存在误报为未授权断言；这些是将节拍写成小说过程所必需的低风险表达。也不要要求每个低风险动作都对应一个事实 ID。

所有当前段应负责的事实 ID 必须完整归入“已实现事实ID”或“缺失事实ID”；所有当前段应负责的事件 ID 必须完整归入“已实现事件ID”或“缺失事件ID”；所有当前段节拍列出的叙事机制 ID 必须完整归入“已实现叙事机制ID”或“缺失叙事机制ID”。同类数组不得重叠。没有被写作包许可的具体断言必须逐条写入“未授权断言”。只有缺失项、未授权断言和问题均为空时，“通过”才可为 true。不要把一般文学偏好、常识补全或来源正文当作依据。

只输出一个合法 JSON object，顶层字段必须且只能是 `段落ID`、`通过`、`已实现事实ID`、`缺失事实ID`、`已实现事件ID`、`缺失事件ID`、`已实现叙事机制ID`、`缺失叙事机制ID`、`未授权断言`、`问题`。数组即使为空也必须保留。JSON 示例：
{json.dumps(example, ensure_ascii=False, indent=2)}

<段落写作包>
{{PARAGRAPH_WRITER_PACKET_JSON}}
</段落写作包>
<待校验段落>
{{PARAGRAPH_DRAFT_JSON}}
</待校验段落>
<本段必须完整核验的ID清单>
{{PARAGRAPH_OBLIGATION_IDS_JSON}}
</本段必须完整核验的ID清单>
"""


def _paragraph_repair_prompt() -> str:
    example = {
        "场景ID": "scene-001",
        "段落正文": [
            {"段落ID": "scene-001:paragraph-01", "正文": "这里是修复后的完整中文小说段落。", "自检兑现事实ID": ["fact-002"], "自检兑现事件ID": ["event-001"]},
        ],
    }
    return f"""# 单段落定点修复提示词

你是中文小说单段定点修复器。只重写 `<待修复段落>` 中唯一的当前段落。依据 `<段落校验结果>` 补齐缺失的既定行动、压力、反应、选择或结果；若问题以“局部正文门槛”开头，必须逐条满足：字数不足就补演当前动作/压力/反应，对话不足就补写有施压和回应的直接对白，缺少动作或感知就把既有节拍演出。对每个 `当前段落节拍`，都应至少呈现一个具体动作、一个会影响视角人物判断的细节，以及由此产生的反应或下一步动作；不能用“告别后离开”“决定去做”替代过程。必须逐项满足 `<本段局部正文门槛>` 的所有数值。不得增加未许可的人物、关系、时间、空间、设定、资源或未来事件。`章节合同窗口` 只允许当前段推进其中的当前事件；`本章段落状态簿` 和 `前文交接` 是已验证前文，不能被改写。不得读取、请求、复述或仿写任何原文。

修复时先逐项执行 `<当前段落节拍>` 的“可见动作”：若事件动作含有“返回、进入、交付、拒绝”等多个动作，不得只保留叹息、解释或结论。愿望、情绪、判断等事实必须借由当前许可人物的动作、视线、感官变化或直接对白落地，不能只改写为“他很想”“他觉得”。“场景定位”段只可修复已成立的场内静态画面，不能偷写尚未绑定事件的移动。

修复后必须是连贯的小说正文而不是梗概。保留当前段落 ID，只输出该段的完整替换文本和实际兑现的 ID。只输出一个合法 JSON object，顶层字段必须且只能是 `场景ID`、`段落正文`；段落项字段必须且只能是 `段落ID`、`正文`、`自检兑现事实ID`、`自检兑现事件ID`。JSON 示例：
{json.dumps(example, ensure_ascii=False, indent=2)}

<段落写作包>
{{PARAGRAPH_WRITER_PACKET_JSON}}
</段落写作包>
<待修复段落>
{{PARAGRAPH_DRAFT_JSON}}
</待修复段落>
<段落校验结果>
{{PARAGRAPH_VALIDATION_JSON}}
</段落校验结果>
<风格执行蓝图>
{{STYLE_EXECUTION_BLUEPRINT_JSON}}
</风格执行蓝图>
<本段局部正文门槛>
{{PARAGRAPH_LOCAL_REQUIREMENTS_JSON}}
</本段局部正文门槛>
""" + _REPAIR_LENGTH_CONTRACT


_EVENT_FRAME_AFFORDANCE = """

事件可执行性必须以语义框架判定，不得从“动作”摘要中搜索个别动词猜测角色：
1. `行动类型` 说明事件的一般语义类别；`actor_id` 是真正发起动作的实体；`target_ids` 只包含必须在当前动作中被作用、接收或回应的实体。
2. `basis_fact_ids` 必须能直接支撑行动者、受事者与动作的关系；只在信息内被提到、却不参与当前动作的实体，不应凭摘要文字被追加为受事者。
3. 当 `行动类型` 为“交流”且 `target_ids` 为空时，只能写成无指定受话者的叙述性交代；不可生成对话往返或虚构听者。当 `target_ids` 非空时，才能让相应受事者参与该动作。
"""


def _scene_affordance_validate_prompt() -> str:
    example = {
        "场景ID": "scene-001",
        "通过": False,
        "不可执行事件ID": ["event-001"],
        "不可执行节拍ID": [],
        "问题": [{
            "对象类型": "事件",
            "对象ID": "event-001",
            "问题": "事件的受事者不在参与者中，且行动依据事实也未建立行动者与该受事者的作用关系。",
            "修复方向": "修正事件语义框架，补齐受事者的参与关系与依据事实，或删除不应参与动作的受事者。",
        }],
    }
    return f"""# 场景可执行性预检提示词

你是中文小说结构预检器。只读取 `<章节实体上下文>` 与 `<当前场景程序>`；禁止索取、输入或依据原文、完整场景正文、未来章节或常识补全。你的任务不是评价文笔，而是判断每个事件和节拍能否仅凭当前程序已有的实体、参与者、事实、时间/空间状态与已许可结果写成正文。

仅在结构确实缺件时判为不可执行：例如事件动作明确要求特定人物、关系角色、物件、地点前提或结果，但当前场景没有相应实体参与、事实或状态可供写作；或节拍的行动者/必要动作没有结构支撑。泛称背景、低风险姿态和氛围不构成缺件。不要因为正文尚未生成而凭空报错，也不要把行动本身的正常展开误判为需要新增事实。

对每个事件逐项检查 `行动类型`、`actor_id`、`target_ids` 与 `basis_fact_ids`。行动者必须是参与者；受事者必须在当前场景可用；依据事实必须支撑这些角色与当前动作。不得因为事件摘要中出现了某个称谓或动词，便自行增加行动角色或专用判定规则。

所有不可执行事件 ID 必须列入 `不可执行事件ID`，所有不可执行节拍 ID 必须列入 `不可执行节拍ID`；每一个 ID 必须恰有一项 `问题` 说明原因和结构性修复方向。`通过` 为 true 时三个数组都必须为空。只输出一个合法 JSON object，顶层字段必须且只能是 `场景ID`、`通过`、`不可执行事件ID`、`不可执行节拍ID`、`问题`；问题项字段必须且只能是 `对象类型`、`对象ID`、`问题`、`修复方向`。JSON 示例：
{json.dumps(example, ensure_ascii=False, indent=2)}

<章节实体上下文>
{{CHAPTER_PROGRAM_CONTEXT_JSON}}
</章节实体上下文>
<当前场景程序>
{{SCENE_PROGRAM_JSON}}
</当前场景程序>
""" + _EVENT_FRAME_AFFORDANCE


def _scene_validate_prompt() -> str:
    example = {
        "场景ID": "scene-001",
        "通过": False,
        "已实现事实ID": ["fact-001"],
        "缺失事实ID": ["fact-002"],
        "已实现事件ID": [],
        "缺失事件ID": ["event-001"],
        "提前泄露事件ID": [],
        "未授权断言段落ID": ["scene-001:paragraph-01"],
        "段落问题": [{"段落ID": "scene-001:paragraph-01", "问题": "结果事实尚未在正文中成立", "修复要求": "在本段补足行动结果，不增加新事件"}],
        "修复段落ID": ["scene-001:paragraph-01"],
    }
    return f"""# 场景语义校验提示词

你是中文小说场景校验器。只比较 `<当前场景程序>`、`<叙事事实子图>` 与 `<场景正文>`；不得根据来源正文、常识或其他场景补充判断。`<叙事事实子图>` 是唯一可读取的跨段事实来源，禁止索取、输入或依据整章原文。逐项核验事实、事件、状态和时间顺序是否已在正文中成立。

所有本场景“必须表达”的事实 ID 和“必现事件”ID 都必须被完整归入“已实现”或“缺失”，且两个数组不能重叠。只有正文已经明确成立的内容才能放入“已实现”。若正文把 `禁止提前事件ID` 中的事件写成已发生、完整预告或倒叙既成事实，必须列入“提前泄露事件ID”。若节拍标注“合理推导”或“氛围扩写”，只能接受其 `依据事实ID` 可直接支撑的动作过程、感知、对白承接或环境节奏。新增人物关系、具体动机、地点转移、明确时间跳转、资源、设定或既成结果不在许可内；但已许可人物不改变持久状态的姿态、普通感知、同场存在与环境细节可作为节拍表达，不应误报。若段落增加本场景程序未许可的持久故事断言，或把其他场景的实体与事实提前写入，必须列入“未授权断言段落ID”，并在“段落问题”说明其内容。`通过` 为 true 时，不得有缺失事实、缺失事件、提前泄露事件、未授权断言或段落问题。问题必须指向可修复的具体段落，修复要求只能要求改写本场景内容。

只输出一个合法 JSON object，顶层字段必须且只能是 `场景ID`、`通过`、`已实现事实ID`、`缺失事实ID`、`已实现事件ID`、`缺失事件ID`、`提前泄露事件ID`、`未授权断言段落ID`、`段落问题`、`修复段落ID`。`未授权断言段落ID` 非空时必须全部列入 `修复段落ID`。`段落问题` 每项必须且只能有 `段落ID`、`问题`、`修复要求`。JSON 示例：
{json.dumps(example, ensure_ascii=False, indent=2)}

<章节实体上下文>
{{CHAPTER_PROGRAM_CONTEXT_JSON}}
</章节实体上下文>
<当前场景程序>
{{SCENE_PROGRAM_JSON}}
</当前场景程序>
<叙事事实子图>
{{NARRATIVE_SUBGRAPH_JSON}}
</叙事事实子图>
<场景正文>
{{SCENE_DRAFT_JSON}}
</场景正文>
"""


def _scene_prose_validate_prompt() -> str:
    example = {
        "场景ID": "scene-001",
        "正文性通过": False,
        "已戏剧化节拍ID": ["scene-001:beat-00"],
        "缺失节拍ID": ["scene-001:beat-01"],
        "摘要化段落ID": ["scene-001:paragraph-01"],
        "段落问题": [{
            "段落ID": "scene-001:paragraph-01",
            "问题": "只交代人物决定，未演出压力和回应",
            "修复要求": "保留既有结果，补写触发选择的可见变化和人物回应",
        }],
        "修复段落ID": ["scene-001:paragraph-01"],
    }
    return f"""# 场景正文性校验提示词

你是独立的中文小说正文性校验器。只比较 `<当前场景程序>`、`<叙事事实子图>` 与 `<场景正文>`，不使用来源正文、常识或其他场景补全判断。`<叙事事实子图>` 是唯一可读取的跨段事实来源，禁止索取、输入或依据整章原文。结构事实是否正确由另一校验器负责；你的任务是判断正文是否把 `戏剧节拍` 演成了小说过程，而非把结果写成剧情摘要。

逐一核验每个节拍：正文必须能看见该节拍要求的 `可见动作`，并使 `压力或阻力`、`视角细节`、`状态变化` 中与该节拍有关的内容产生因果联系。只有“某人得知、决定、发现、说出、于是、最终”的结果句，没有动作、压力、反应或选择过程，属于摘要化。对白只是在说明设定、没有施压和回应，也属于摘要化。可感知细节若不影响人物注意力、判断或动作，不能算作节拍已戏剧化。

所有 `戏剧节拍` ID 必须完整归入 `已戏剧化节拍ID` 或 `缺失节拍ID`，两者不得重叠。只要存在缺失节拍、摘要化段落或段落问题，`正文性通过` 必须为 false。每个问题必须定位到可修复段落，修复要求只能要求重演既有过程，不能添加新人物、新事实、新事件或未来情节。

只输出一个合法 JSON object，顶层字段必须且只能是 `场景ID`、`正文性通过`、`已戏剧化节拍ID`、`缺失节拍ID`、`摘要化段落ID`、`段落问题`、`修复段落ID`。`段落问题` 每项必须且只能有 `段落ID`、`问题`、`修复要求`。JSON 示例：
{json.dumps(example, ensure_ascii=False, indent=2)}

<章节实体上下文>
{{CHAPTER_PROGRAM_CONTEXT_JSON}}
</章节实体上下文>
<当前场景程序>
{{SCENE_PROGRAM_JSON}}
</当前场景程序>
<叙事事实子图>
{{NARRATIVE_SUBGRAPH_JSON}}
</叙事事实子图>
<场景正文>
{{SCENE_DRAFT_JSON}}
</场景正文>
"""


def _scene_repair_prompt() -> str:
    example = {
        "场景ID": "scene-001",
        "修复段落": [
            {"段落ID": "scene-001:paragraph-01", "正文": "这里是替换后的完整段落。", "自检兑现事实ID": ["fact-002"], "自检兑现事件ID": ["event-001"]},
        ],
    }
    return f"""# 场景定点段落修复提示词

修复时同时读取 `<场景正文性校验结果>`：若其中指出摘要化或缺失节拍，必须依据相应节拍重演可见动作、压力、人物反应或取舍，不得只把“得知、决定、发生、于是、最终”等概述词换成更长的说明。对白必须带有目的、压力或回应；细节必须服务人物下一步的判断或动作。

你是中文小说定点修复器。只重写 `<待修复段落ID>` 指定的段落；不得输出、改写、删除或合并其他段落。保留每个段落 ID，并根据 `<场景校验结果>` 补齐缺失事实、缺失事件、状态或因果。`<叙事事实子图>` 是唯一可读取的跨段事实来源，禁止索取、输入或依据整章原文。若 `<局部风格修复要求>` 非空，同时只在这些目标段内调整句段节奏，并落实该段在 `<风格执行蓝图>` 中的 `职责标签` 与 `叙事职责`。这些职责会被程序逐段检测：对话职责必须以中文引号“”写出不少于 `最低对白字数` 的直接对白；转场职责必须出现指定承接词之一；感知职责必须出现指定感知词之一；短段职责不得超过该段 `最大字数`。对话只能发生在当前场景已列出的参与者之间；转场和感知只能承接已有的状态、行动与事实。不得增加新事实，不得改变已成立事实，不得写入禁止提前事件。

输出的是可直接替换原段落的完整中文正文，不输出解释、标题、Markdown 或其他段落。

只输出一个合法 JSON object，顶层字段必须且只能是 `场景ID`、`修复段落`。`修复段落` 中每项字段必须且只能是 `段落ID`、`正文`、`自检兑现事实ID`、`自检兑现事件ID`，且必须恰好覆盖所有待修复段落 ID。`正文` 必须是单行 JSON 字符串；小说对白只能使用中文引号“”，不得在正文中使用未转义的 ASCII 双引号。禁止输出空对象 `{{}}`、空数组或省略任一顶层字段；若内容很短，也必须完整填写下面示例中的字段结构。JSON 示例：
{json.dumps(example, ensure_ascii=False, indent=2)}

<章节实体上下文>
{{CHAPTER_PROGRAM_CONTEXT_JSON}}
</章节实体上下文>
<当前场景程序>
{{SCENE_PROGRAM_JSON}}
</当前场景程序>
<叙事事实子图>
{{NARRATIVE_SUBGRAPH_JSON}}
</叙事事实子图>
<完整场景正文>
{{SCENE_DRAFT_JSON}}
</完整场景正文>
<场景校验结果>
{{SCENE_VALIDATION_JSON}}
</场景校验结果>
<场景正文性校验结果>
{{SCENE_PROSE_VALIDATION_JSON}}
</场景正文性校验结果>
<待修复段落ID>
{{TARGET_PARAGRAPH_IDS_JSON}}
</待修复段落ID>
<风格执行蓝图>
{{STYLE_EXECUTION_BLUEPRINT_JSON}}
</风格执行蓝图>
<局部风格修复要求>
{{STYLE_REPAIR_DIRECTIVES_JSON}}
</局部风格修复要求>
"""


def _validate_prompt() -> str:
    example = {
        "passed": False,
        "issues": [{"code": "state_jump", "severity": "error", "evidence": "第2场景的位置变化没有计划依据", "repair": "补写移动原因并更新状态"}],
        "style_observations": ["短段与完整段交替已出现"],
        "style_budget_result": {"passed": True, "notes": ["主要节奏指标大致符合计划"]},
        "repairs": ["先修复状态跳变，再检查正文节奏"],
    }
    return f"""# 章节一致性与提示词执行校验提示词

你是小说章节校验器。遵循 `<ExecutionMode>` 判定哪些输入事实可被正文使用；只比较允许使用的故事状态、章节计划和正文。检查实体、关系、时间、位置、资源、知识、因果、场景进入/退出状态，以及模板和文风约束是否被合理执行。无姓名、仅承担叙述指代功能的泛称（如“族中长辈”“路人”“一名弟子”）不是新增实体；只有被赋予独立身份、关系、行动或状态的特定人物才可判为实体臆造。保真评测模式下，`<ChapterPlan>.scene_contracts` 中每场的事件 `action`、`outcomes`、`costs`、`entry_facts` 和 `exit_facts` 都是硬条件：正文漏写、颠倒因果、把关系/态度改成相反含义，必须输出 `severity: "error"` 的 issue；不得因为正文流畅就判通过。`baselines` 是柔性范围，不可把偏离一次范围当作错误。

只输出一个合法 JSON object，顶层字段必须且只能是：`passed`、`issues`、`style_observations`、`style_budget_result`、`repairs`。`issues` 中每项必须含 `code`、`severity`、`evidence`、`repair`；`style_budget_result` 说明本章预算是否大致执行，但不能把一次偏离柔性区间单独视为逻辑错误。

JSON 示例：
{json.dumps(example, ensure_ascii=False, indent=2)}

<AuthorStyleProfile>
{{AUTHOR_STYLE_PROFILE_JSON}}
</AuthorStyleProfile>
<ContinuityRequirements>
{{CONTINUITY_REQUIREMENTS_JSON}}
</ContinuityRequirements>
<StyleExecutionSpec>
{{STYLE_EXECUTION_SPEC_JSON}}
</StyleExecutionSpec>
<ExecutionMode>
{{EXECUTION_MODE_INSTRUCTIONS}}
</ExecutionMode>
<OriginalStoryState>
{{ORIGINAL_STORY_STATE_JSON}}
</OriginalStoryState>
<ChapterPlan>
{{CHAPTER_PLAN_JSON}}
</ChapterPlan>
<Draft>
{{DRAFT_TEXT}}
</Draft>
"""


def _skill_markdown(skill_name: str = "novel-author-skill") -> str:
    """Compatibility wrapper for tests and callers of the old helper."""

    return build_skill_markdown(skill_name)


def _assess_inputs(
    profile: AuthorStyleProfile,
    library: NarrativeTemplateLibrary,
    continuity: WorkContinuity,
    source_terms: Iterable[str],
) -> tuple[SkillPackageIssue, ...]:
    issues: list[SkillPackageIssue] = []
    for name, object_ in (("style_profile", profile), ("template_library", library), ("continuity", continuity)):
        try:
            object_.validate()
        except (TypeError, ValueError) as exc:
            issues.append(SkillPackageIssue("invalid_input_contract", f"{name}: {exc}"))
    if profile.author_id != library.author_id or profile.author_id != continuity.author_id:
        issues.append(SkillPackageIssue("author_mismatch", "三个阶段产物的 author_id 不一致"))
    if profile.work_id != library.work_id or profile.work_id != continuity.work_id:
        issues.append(SkillPackageIssue("work_mismatch", "三个阶段产物的 work_id 不一致"))
    if profile.chapter_ids != library.chapter_ids or profile.chapter_ids != continuity.chapter_ids:
        issues.append(SkillPackageIssue("chapter_sample_mismatch", "三个阶段产物的章节样本不一致"))
    style_report = assess_style_profile(profile, source_terms)
    for item in style_report.issues:
        if item.severity == "error":
            issues.append(SkillPackageIssue("invalid_style_profile", item.message))
    template_report = assess_template_library(library)
    for item in template_report.issues:
        if item.severity == "error":
            issues.append(SkillPackageIssue("invalid_template_library", item.message))
    source_terms = tuple(source_terms)
    for value in _template_texts(library):
        if any(term in value for term in source_terms):
            issues.append(SkillPackageIssue("source_term_in_template", "结构模板含有来源特定名称"))
            break
    expected = _CORE_FAMILIES - {item.family for item in profile.constraints}
    if expected:
        issues.append(SkillPackageIssue("missing_core_style_family", f"缺少核心文风类别：{', '.join(sorted(expected))}"))
    return tuple(issues)


def _semantic_audit_evidence(folder: Path) -> dict[str, Any]:
    """Load the largest available semantic-audit summary without source prose."""

    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    invalid: list[str] = []
    for path in folder.glob("semantic_audit.sample-*.quality.json"):
        try:
            payload = _read_json(path)
            count = int(payload.get("content_audit_chapter_count", 0))
            if count < 0 or payload.get("pass_scope") != "model_content_semantic_audit":
                raise ValueError("invalid semantic-audit scope or chapter count")
            candidates.append((count, path, payload))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            invalid.append(f"{path.name}: {type(exc).__name__}: {exc}")
    if not candidates:
        return {
            "available": False,
            "artifact": "",
            "content_audit_chapter_count": 0,
            "passed": False,
            "issue_count": 0,
            "error_count": 0,
            "invalid_artifacts": invalid,
        }
    count, path, payload = sorted(candidates, key=lambda item: (item[0], item[1].name))[-1]
    return {
        "available": True,
        "artifact": path.name,
        "report": str(payload.get("report", "")),
        "content_audit_chapter_count": count,
        "batch_count": int(payload.get("batch_count", 0)),
        "issue_count": int(payload.get("issue_count", 0)),
        "error_count": int(payload.get("error_count", 0)),
        "passed": bool(payload.get("passed", False)) and count >= 5,
        "invalid_artifacts": invalid,
    }


def _qualification(audit: dict[str, Any], *, portable_self_check_passed: bool = False) -> SkillQualification:
    return SkillQualification(
        source_work_count=1,
        content_audit_chapter_count=int(audit.get("content_audit_chapter_count", 0)),
        semantic_audit_passed=bool(audit.get("passed", False)),
        portable_self_check_passed=portable_self_check_passed,
    )


def _draft_release_issues(audit: dict[str, Any]) -> tuple[SkillPackageIssue, ...]:
    """Explain why a work-scoped compiler output cannot claim author validity."""

    if not audit.get("available"):
        semantic_issue = SkillPackageIssue("semantic_audit_missing", "尚无随机五章内容级语义审计记录。", "warning")
    elif int(audit.get("content_audit_chapter_count", 0)) < 5:
        semantic_issue = SkillPackageIssue(
            "semantic_audit_insufficient",
            f"内容级语义审计仅覆盖 {int(audit.get('content_audit_chapter_count', 0))} 章，发布门槛至少为 5 章。",
            "warning",
        )
    elif not audit.get("passed"):
        semantic_issue = SkillPackageIssue(
            "semantic_audit_failed",
            f"五章内容级语义审计尚未通过：{int(audit.get('error_count', 0))} 个错误，{int(audit.get('issue_count', 0))} 个总问题。",
            "warning",
        )
    else:
        semantic_issue = None
    result = [
        SkillPackageIssue("single_work_only", "当前包只含一部作品，候选规律不能进入跨作品作者核心。", "warning"),
        SkillPackageIssue("genre_contrast_missing", "尚无同题材对照基线，不能证明规则具有作者区分度。", "warning"),
        SkillPackageIssue("held_out_evaluation_missing", "尚无留出作品、原创长篇和防复写发布验收。", "warning"),
    ]
    if semantic_issue is not None:
        result.insert(1, semantic_issue)
    if audit.get("invalid_artifacts"):
        result.append(SkillPackageIssue("semantic_audit_artifact_invalid", "部分语义审计产物无法读取，已忽略。", "warning"))
    return tuple(result)


def compile_author_skill(
    author_id: str,
    work_id: str,
    *,
    limit: int = DEFAULT_SKILL_LIMIT,
) -> tuple[Path, Path]:
    """Compile work-scoped artifacts into an honest portable draft Skill."""

    folder = corpus_dir(author_id, work_id)
    profile_path = _artifact_path(folder, "author_style_profile", limit, ".json")
    template_path = _artifact_path(folder, "narrative_templates", limit, ".json")
    continuity_path = _artifact_path(folder, "work_continuity", limit, ".json")
    profile = AuthorStyleProfile.from_dict(_read_json(profile_path))
    library = NarrativeTemplateLibrary.from_dict(_read_json(template_path))
    continuity = WorkContinuity.from_dict(_read_json(continuity_path))
    semantic_audit = _semantic_audit_evidence(folder)
    source_terms = _source_terms(folder, limit)
    issues = _assess_inputs(profile, library, continuity, source_terms)
    if any(item.severity == "error" for item in issues):
        messages = "; ".join(f"{item.code}: {item.message}" for item in issues)
        raise ValueError(f"cannot compile skill: {messages}")

    destination = output_dir(author_id)
    destination.mkdir(parents=True, exist_ok=True)
    for name in _LEGACY_OUTPUT_FILES:
        path = destination / name
        if path.exists():
            path.unlink()
    generated_directories = ("agents", "references", "scripts", "evaluation", "evaluation_rules", "prompt_templates")
    for name in generated_directories:
        path = destination / name
        if path.exists():
            shutil.rmtree(path)

    skill_name = normalize_skill_name(author_id)
    prompt_dir = destination / "references" / "prompts"
    prompt_dir.mkdir(parents=True)
    (destination / "agents").mkdir()
    (destination / "scripts").mkdir()
    (destination / "evaluation").mkdir()

    requirements = _continuity_requirements(continuity)
    execution_spec = style_execution_spec(profile)
    references = build_draft_references(profile, library, continuity, execution_spec)
    evidence_report = references["references/evidence_report.json"]
    evidence_report["content_semantic_audit"] = semantic_audit
    missing_gates = list(evidence_report.get("missing_evidence_gates", []))
    if semantic_audit.get("passed"):
        missing_gates = [item for item in missing_gates if item != "五章内容级语义审计结果"]
    evidence_report["missing_evidence_gates"] = missing_gates
    references.update({
        "references/author_style_profile.json": profile.to_dict(),
        "references/narrative_templates.json": library.to_dict(),
        "references/continuity_requirements.json": requirements,
        "references/style_execution_spec.json": execution_spec,
    })

    prompt_program = {
        "schema_version": "2.0",
        "release_status": "draft",
        "stages": [
            {"stage": "planning", "template": "references/prompts/chapter_plan.md", "required_inputs": ["author_style_profile", "narrative_templates", "continuity_requirements", "style_execution_spec", "original_story_state", "chapter_brief"], "output": "ChapterPlan JSON with style_budget"},
            {"stage": "drafting", "template": "references/prompts/chapter_draft.md", "required_inputs": ["author_style_profile", "selected_templates", "continuity_requirements", "style_execution_spec", "original_story_state", "chapter_plan"], "output": "chapter prose"},
            {"stage": "validation", "template": "references/prompts/chapter_validate.md", "required_inputs": ["author_style_profile", "continuity_requirements", "style_execution_spec", "original_story_state", "chapter_plan", "draft"], "output": "ValidationReport JSON"},
            {"stage": "scene_drafting", "template": "references/prompts/scene_draft.md", "required_inputs": ["author_style_profile", "selected_templates", "style_execution_blueprint", "paragraph_writer_packet"], "output": "单段落ID对齐的正文JSON"},
            {"stage": "paragraph_validation", "template": "references/prompts/paragraph_validate.md", "required_inputs": ["paragraph_writer_packet", "paragraph_draft"], "output": "ParagraphValidation JSON"},
            {"stage": "paragraph_repair", "template": "references/prompts/paragraph_repair.md", "required_inputs": ["paragraph_writer_packet", "paragraph_draft", "paragraph_validation", "style_execution_blueprint"], "output": "当前单段的修复JSON"},
            {"stage": "scene_affordance_validation", "template": "references/prompts/scene_affordance_validate.md", "required_inputs": ["chapter_program_context", "scene_program"], "output": "SceneAffordanceValidation JSON"},
            {"stage": "scene_validation", "template": "references/prompts/scene_validate.md", "required_inputs": ["chapter_program_context", "scene_program", "narrative_subgraph", "scene_draft"], "output": "SceneValidation JSON"},
            {"stage": "scene_prose_validation", "template": "references/prompts/scene_prose_validate.md", "required_inputs": ["chapter_program_context", "scene_program", "narrative_subgraph", "scene_draft"], "output": "SceneProseValidation JSON"},
            {"stage": "scene_repair", "template": "references/prompts/scene_repair.md", "required_inputs": ["chapter_program_context", "scene_program", "narrative_subgraph", "scene_draft", "scene_validation", "scene_prose_validation", "target_paragraph_ids"], "output": "仅目标段落的修复JSON"},
        ],
        "repair_loop": "任一事实、事件、时序、状态或正文性硬校验失败时，只修订指定段落并重新校验；未通过的候选不得发布。",
    }
    references["references/prompt_program.json"] = prompt_program

    generated = (
        "SKILL.md", "agents/openai.yaml", "skill_manifest.json", "skill_quality.json",
        "scripts/self_check.py", "scripts/select_style_context.py", "evaluation/package_self_check.json",
        "evaluation/content_semantic_audit.json",
        *tuple(references),
        "references/prompts/chapter_plan.md", "references/prompts/chapter_draft.md", "references/prompts/chapter_validate.md",
        "references/prompts/scene_draft.md", "references/prompts/paragraph_validate.md", "references/prompts/paragraph_repair.md",
        "references/prompts/scene_affordance_validate.md", "references/prompts/scene_validate.md",
        "references/prompts/scene_prose_validate.md", "references/prompts/scene_repair.md",
    )
    (destination / "SKILL.md").write_text(_skill_markdown(skill_name), encoding="utf-8")
    (destination / "agents" / "openai.yaml").write_text(build_openai_yaml(skill_name, author_id), encoding="utf-8")
    (destination / "scripts" / "self_check.py").write_text(self_check_script(), encoding="utf-8")
    (destination / "scripts" / "select_style_context.py").write_text(select_style_context_script(), encoding="utf-8")
    for relative, payload in references.items():
        write_json(destination / relative, payload)
    write_json(destination / "evaluation" / "content_semantic_audit.json", {
        "schema_version": "1.0",
        "pass_scope": "model_content_semantic_audit_summary",
        **semantic_audit,
    })
    (prompt_dir / "chapter_plan.md").write_text(_plan_prompt(), encoding="utf-8")
    (prompt_dir / "chapter_draft.md").write_text(_draft_prompt(), encoding="utf-8")
    (prompt_dir / "chapter_validate.md").write_text(_validate_prompt(), encoding="utf-8")
    (prompt_dir / "scene_draft.md").write_text(_scene_draft_prompt(), encoding="utf-8")
    (prompt_dir / "paragraph_validate.md").write_text(_paragraph_validate_prompt(), encoding="utf-8")
    (prompt_dir / "paragraph_repair.md").write_text(_paragraph_repair_prompt(), encoding="utf-8")
    (prompt_dir / "scene_affordance_validate.md").write_text(_scene_affordance_validate_prompt(), encoding="utf-8")
    (prompt_dir / "scene_validate.md").write_text(_scene_validate_prompt(), encoding="utf-8")
    (prompt_dir / "scene_prose_validate.md").write_text(_scene_prose_validate_prompt(), encoding="utf-8")
    (prompt_dir / "scene_repair.md").write_text(_scene_repair_prompt(), encoding="utf-8")

    limitations = (
        "当前仅有单作品证据，作者核心为空。",
        "尚未建立同题材对照基线。",
        "尚未通过随机五章语义审计、留出作品、原创长篇和防复写验收。",
    )
    qualification = _qualification(semantic_audit)
    bundle = AuthorSkillBundle(
        author_id=author_id,
        skill_name=skill_name,
        release_status="draft",
        source_work_ids=(work_id,),
        chapter_ids=profile.chapter_ids,
        candidate_trait_count=len(profile.constraints),
        validated_author_trait_count=0,
        narrative_policy_count=len(library.templates),
        generated_files=generated,
        limitations=limitations,
        qualification=qualification,
    )
    bundle.validate()

    def write_manifest(current: AuthorSkillBundle) -> None:
        write_json(destination / "skill_manifest.json", {
            **current.to_dict(),
            "work_id": work_id,
            "chapter_count": len(profile.chapter_ids),
            "template_count": len(library.templates),
            "style_constraint_count": len(profile.constraints),
            "style_budget_metric_count": len(execution_spec["metrics"]),
            "source_artifacts": {
                "continuity": continuity_path.name,
                "templates": template_path.name,
                "style": profile_path.name,
                "semantic_audit_quality": semantic_audit.get("artifact", ""),
            },
        })

    write_manifest(bundle)
    pending = {"skill_quality.json", "evaluation/package_self_check.json"}
    package_issues = assess_skill_directory(destination, (name for name in generated if name not in pending))
    qualification = _qualification(semantic_audit, portable_self_check_passed=not package_issues)
    bundle = AuthorSkillBundle(
        author_id=bundle.author_id,
        skill_name=bundle.skill_name,
        release_status=bundle.release_status,
        source_work_ids=bundle.source_work_ids,
        chapter_ids=bundle.chapter_ids,
        candidate_trait_count=bundle.candidate_trait_count,
        validated_author_trait_count=bundle.validated_author_trait_count,
        narrative_policy_count=bundle.narrative_policy_count,
        generated_files=bundle.generated_files,
        limitations=bundle.limitations,
        qualification=qualification,
    )
    bundle.validate()
    write_manifest(bundle)
    write_json(destination / "evaluation" / "package_self_check.json", {
        "schema_version": "1.0",
        "passed": not package_issues,
        "package_root": ".",
        "checked_file_count": len(generated) - len(pending),
        "issues": list(package_issues),
    })

    quality_issues = [*issues, *_draft_release_issues(semantic_audit)]
    quality_issues.extend(SkillPackageIssue("package_self_check", item) for item in package_issues)
    report = SkillPackageQualityReport(
        author_id=author_id,
        work_id=work_id,
        chapter_count=len(profile.chapter_ids),
        template_count=len(library.templates),
        style_constraint_count=len(profile.constraints),
        generated_files=generated,
        release_status="draft",
        qualification=qualification,
        issues=tuple(quality_issues),
    )
    report_path = destination / "skill_quality.json"
    write_json(report_path, report.to_dict())

    final_issues = assess_skill_directory(destination, generated)
    if final_issues:
        known = {(item.code, item.message) for item in report.issues}
        additions = tuple(
            SkillPackageIssue("package_self_check", item)
            for item in final_issues if ("package_self_check", item) not in known
        )
        if additions:
            report = SkillPackageQualityReport(
                author_id=report.author_id,
                work_id=report.work_id,
                chapter_count=report.chapter_count,
                template_count=report.template_count,
                style_constraint_count=report.style_constraint_count,
                generated_files=report.generated_files,
                release_status=report.release_status,
                qualification=_qualification(semantic_audit, portable_self_check_passed=False),
                issues=(*report.issues, *additions),
            )
            write_json(report_path, report.to_dict())
            write_json(destination / "evaluation" / "package_self_check.json", {
                "schema_version": "1.0", "passed": False, "package_root": ".",
                "checked_file_count": len(generated), "issues": list(final_issues),
            })
    if not report.passed:
        raise ValueError("compiled skill did not pass package quality checks")
    return destination, report_path
