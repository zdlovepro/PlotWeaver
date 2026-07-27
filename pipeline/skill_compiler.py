"""Stage 7: compile evidence-first artifacts into a standalone prompt skill.

The package deliberately contains abstract templates, quantified style ranges and
continuity requirements only.  It does not copy source prose, source names, or a
work-specific story state into prompts for new writing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from .contracts import AuthorStyleProfile, ChapterAnnotation, NarrativeTemplateLibrary, WorkContinuity
from .jsonio import read_jsonl, write_json
from .paths import corpus_dir, output_dir
from .style_distillation import assess_style_profile
from .style_execution import style_execution_spec
from .template_mining import assess_template_library


DEFAULT_SKILL_LIMIT = 20
_CORE_FAMILIES = frozenset({"sentence_rhythm", "paragraph_pacing", "dialogue", "transition", "focalization"})
_LEGACY_OUTPUT_FILES = (
    "author_narrative_profile.json", "chapter_state_schema.json", "distillation_batches.jsonl",
    "skill_context.json", "template_library.json", "template_mining_batches.jsonl",
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
    issues: tuple[SkillPackageIssue, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not any(item.severity == "error" for item in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "author_id": self.author_id,
            "work_id": self.work_id,
            "chapter_count": self.chapter_count,
            "template_count": self.template_count,
            "style_constraint_count": self.style_constraint_count,
            "generated_files": list(self.generated_files),
            "passed": self.passed,
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


def _scene_draft_prompt() -> str:
    example = {
        "场景ID": "scene-001",
        "段落正文": [
            {"段落ID": "scene-001:paragraph-00", "正文": "这里是完整的中文小说段落。", "自检兑现事实ID": ["fact-001"], "自检兑现事件ID": []},
            {"段落ID": "scene-001:paragraph-01", "正文": "这里推进动作并呈现结果。", "自检兑现事实ID": ["fact-002"], "自检兑现事件ID": ["event-001"]},
        ],
    }
    return f"""# 场景—段落正文生成提示词

你是中文小说作者。只根据给定的实体字典、当前场景程序、叙事事实子图和抽象风格约束，逐段生成当前场景；不得书写其他场景，不得使用未提供的专名、设定、人物或情节，不得复制、改写或补全任何来源正文句子。`<叙事事实子图>` 是唯一可读取的跨段事实来源：它只包含当前段落获准使用的结构化节点、关系、因果、伏笔状态和已提交状态；禁止索取、输入、引用或依据整章原文。

这里要求的是小说正文，不是剧情梗概。`戏剧节拍` 是每段必须被演出的最小过程：先让读者看见行动或局面变化，再写人物受到的压力、感知或回应，最后才让状态变化成立。不得把关键节拍压缩成“甲得知消息后决定离开”“甲说出消息，于是众人同意”这类结果句。下面是匿名的写法对照：

不合格：甲得知消息后，决定离开。

合格：甲先从对方的停顿、动作或一句对白中确认消息；消息造成的压力使他的注意力或身体反应发生变化；他再以可见动作、对白或取舍落实离开的选择。

每个段落程序绑定的 `节拍ID列表` 都必须在该段中落实其 `可见动作`、`视角细节`、`压力或阻力` 和 `状态变化`。没有压力的节拍也必须让动作造成局面变化；有 `对白压力` 的节拍必须写出施压与回应，且回应影响下一步行动。感知和环境细节必须改变人物的注意力、判断或动作，不能只堆砌形容词。一个段落只展开自己的节拍和一至两个紧密事实，不能把后续结果提前概述。

当前场景中的每个 `段落程序` 都必须恰好生成一段 `正文`，并保留原有 `段落ID`。正文要落实该段的“段落功能”、必须表达的事实 ID 和事件 ID；事实可以通过叙述、行动、对话或心理呈现，但不得改变谓词方向、人物关系、状态结果或因果顺序。`禁止提前事件ID` 中的事件不得在本场景被完整写出、预告为既成事实或回顾为已经发生。

若 `<章节实体上下文>` 的 `生成模式` 为“受控扩写”，必须同时执行每个戏剧节拍的 `扩写许可` 和 `依据事实ID`：

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
<章节实体上下文>
{{CHAPTER_PROGRAM_CONTEXT_JSON}}
</章节实体上下文>
<当前场景程序>
{{SCENE_PROGRAM_JSON}}
</当前场景程序>
<叙事事实子图>
{{NARRATIVE_SUBGRAPH_JSON}}
</叙事事实子图>
"""


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

所有本场景“必须表达”的事实 ID 和“必现事件”ID 都必须被完整归入“已实现”或“缺失”，且两个数组不能重叠。只有正文已经明确成立的内容才能放入“已实现”。若正文把 `禁止提前事件ID` 中的事件写成已发生、完整预告或倒叙既成事实，必须列入“提前泄露事件ID”。若节拍标注“合理推导”或“氛围扩写”，只能接受其 `依据事实ID` 可直接支撑的动作过程、感知、对白承接或环境节奏；任何新增人物关系、具体动机、时间、地点、资源、设定或既成结果都不在许可内。若段落增加本场景程序未许可的具体断言，或把其他场景的实体与事实提前写入，必须列入“未授权断言段落ID”，并在“段落问题”说明其内容。`通过` 为 true 时，不得有缺失事实、缺失事件、提前泄露事件、未授权断言或段落问题。问题必须指向可修复的具体段落，修复要求只能要求改写本场景内容。

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


def _skill_markdown() -> str:
    return """---
name: novel-author-skill
description: 使用已蒸馏的抽象叙事模板、文风约束和连续性规则，规划、生成并校验原创中文小说章节。
---

# 小说作者 Skill

1. 读取 `author_style_profile.json`、`style_execution_spec.json` 与 `narrative_templates.json`。前两者是可度量但非机械凑数的文风预算，模板只能提供抽象叙事功能，不能携带来源角色、术语、世界或情节。
2. 原创写作时，用 `chapter_plan.md` 建立严格 JSON 章节计划，再用 `chapter_draft.md` 和 `chapter_validate.md` 生成、校验和修复。只有 `reconstruction_mode: fidelity_test_only` 才可使用已提供的结构要素，且仍不得输入或复用原文。
3. 保真测试时，先由章节程序提供实体、事实、事件、场景、叙事节拍和段落职责；每一场景使用 `scene_draft.md` 生成按段落 ID 对齐的正文 JSON，不得以剧情梗概代替行动、反应和结果。
4. 长篇正文使用 `controlled_expansion` 章节程序：先设定章节目标字数，再由程序将其分配到场景、段落和节拍。每个非原文节拍必须带有“合理推导”或“氛围扩写”许可及依据事实 ID；不得将扩写当作新增剧情许可。
5. `narrative_graph.json` 是保真生成的唯一跨段事实底座。每次只生成一个段落，并只读取该段的图谱子图：获准实体、事实、人物关系、事件因果、伏笔生命周期和已提交状态；绝不向写作模型输入原文、整章正文或上一段正文尾部。
6. 对每一场景依次运行 `scene_validate.md`（事实、事件、时序与未授权具体断言）和 `scene_prose_validate.md`（每个叙事节拍是否真正演出）。只修复校验明确指出的段落，并重新运行两种校验。
7. 每章完成后先写 `chapter_graph_patch.candidate.json`，其中记录已实现事实、事件因果和状态变化。只有正文、人物状态、事件因果、文风和长度均通过校验，才提交该补丁并让下一章读取它。
8. `candidate_draft.txt` 是待验候选稿。只有全部硬门槛和图谱补丁校验通过，系统才写出 `generated_draft.txt`；未通过时不得把候选稿当成正式正文，也不得开始下一章。
9. `continuity_requirements.json` 只定义校验框架和观测规模，不能被当作新故事的实体或状态数据。
"""


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


def compile_author_skill(
    author_id: str,
    work_id: str,
    *,
    limit: int = DEFAULT_SKILL_LIMIT,
) -> tuple[Path, Path]:
    """Compile stage 4–6 artifacts into ``output/<AuthorId>_skill``."""

    folder = corpus_dir(author_id, work_id)
    profile = AuthorStyleProfile.from_dict(_read_json(_artifact_path(folder, "author_style_profile", limit, ".json")))
    library = NarrativeTemplateLibrary.from_dict(_read_json(_artifact_path(folder, "narrative_templates", limit, ".json")))
    continuity = WorkContinuity.from_dict(_read_json(_artifact_path(folder, "work_continuity", limit, ".json")))
    source_terms = _source_terms(folder, limit)
    issues = _assess_inputs(profile, library, continuity, source_terms)
    if any(item.severity == "error" for item in issues):
        messages = "; ".join(f"{item.code}: {item.message}" for item in issues)
        raise ValueError(f"cannot compile skill: {messages}")

    destination = output_dir(author_id)
    destination.mkdir(parents=True, exist_ok=True)
    # These were produced by the retired state-extractor compiler.  The current
    # package is self-contained, and removing only these known generated paths
    # prevents consumers from mixing incompatible prompt programs.
    for name in _LEGACY_OUTPUT_FILES:
        path = destination / name
        if path.exists():
            path.unlink()
    legacy_dirs = (destination / "evaluation_rules", destination / "prompt_templates")
    for path in legacy_dirs:
        if path.exists():
            shutil.rmtree(path)

    prompt_dir = destination / "prompt_templates"
    prompt_dir.mkdir()
    generated = (
        "SKILL.md", "author_style_profile.json", "narrative_templates.json", "continuity_requirements.json", "style_execution_spec.json",
        "prompt_program.json", "skill_manifest.json", "prompt_templates/chapter_plan.md",
        "prompt_templates/chapter_draft.md", "prompt_templates/chapter_validate.md",
        "prompt_templates/scene_draft.md", "prompt_templates/scene_validate.md", "prompt_templates/scene_prose_validate.md", "prompt_templates/scene_repair.md",
    )
    (destination / "SKILL.md").write_text(_skill_markdown(), encoding="utf-8")
    write_json(destination / "author_style_profile.json", profile.to_dict())
    write_json(destination / "narrative_templates.json", library.to_dict())
    requirements = _continuity_requirements(continuity)
    write_json(destination / "continuity_requirements.json", requirements)
    execution_spec = style_execution_spec(profile)
    write_json(destination / "style_execution_spec.json", execution_spec)
    (prompt_dir / "chapter_plan.md").write_text(_plan_prompt(), encoding="utf-8")
    (prompt_dir / "chapter_draft.md").write_text(_draft_prompt(), encoding="utf-8")
    (prompt_dir / "chapter_validate.md").write_text(_validate_prompt(), encoding="utf-8")
    (prompt_dir / "scene_draft.md").write_text(_scene_draft_prompt(), encoding="utf-8")
    (prompt_dir / "scene_validate.md").write_text(_scene_validate_prompt(), encoding="utf-8")
    (prompt_dir / "scene_prose_validate.md").write_text(_scene_prose_validate_prompt(), encoding="utf-8")
    (prompt_dir / "scene_repair.md").write_text(_scene_repair_prompt(), encoding="utf-8")
    write_json(destination / "prompt_program.json", {
        "schema_version": "1.0",
        "stages": [
            {"stage": "planning", "template": "prompt_templates/chapter_plan.md", "required_inputs": ["author_style_profile", "narrative_templates", "continuity_requirements", "style_execution_spec", "original_story_state", "chapter_brief"], "output": "ChapterPlan JSON with style_budget"},
            {"stage": "drafting", "template": "prompt_templates/chapter_draft.md", "required_inputs": ["author_style_profile", "selected_templates", "continuity_requirements", "style_execution_spec", "original_story_state", "chapter_plan"], "output": "chapter prose"},
            {"stage": "validation", "template": "prompt_templates/chapter_validate.md", "required_inputs": ["author_style_profile", "continuity_requirements", "style_execution_spec", "original_story_state", "chapter_plan", "draft"], "output": "ValidationReport JSON"},
            {"stage": "scene_drafting", "template": "prompt_templates/scene_draft.md", "required_inputs": ["author_style_profile", "selected_templates", "style_execution_blueprint", "chapter_program_context", "scene_program", "narrative_subgraph"], "output": "按段落 ID 对齐的场景正文 JSON"},
            {"stage": "scene_validation", "template": "prompt_templates/scene_validate.md", "required_inputs": ["chapter_program_context", "scene_program", "narrative_subgraph", "scene_draft"], "output": "SceneValidation JSON"},
            {"stage": "scene_prose_validation", "template": "prompt_templates/scene_prose_validate.md", "required_inputs": ["chapter_program_context", "scene_program", "narrative_subgraph", "scene_draft"], "output": "SceneProseValidation JSON"},
            {"stage": "paragraph_repair", "template": "prompt_templates/scene_repair.md", "required_inputs": ["chapter_program_context", "scene_program", "narrative_subgraph", "scene_draft", "scene_validation", "scene_prose_validation", "target_paragraph_ids"], "output": "仅目标段落的修复 JSON"},
        ],
        "repair_loop": "任一场景事实/事件/时序/未授权断言校验或正文性校验未通过时，只修订其指定段落，并重新运行两种校验；未通过的候选稿不得发布。",
    })
    write_json(destination / "skill_manifest.json", {
        "schema_version": "1.0",
        "author_id": author_id,
        "work_id": work_id,
        "chapter_count": len(profile.chapter_ids),
        "template_count": len(library.templates),
        "style_constraint_count": len(profile.constraints),
        "style_budget_metric_count": len(execution_spec["metrics"]),
        "generated_files": list(generated),
        "source_artifacts": {
            "continuity": f"work_continuity.sample-{len(profile.chapter_ids)}.json",
            "templates": f"narrative_templates.sample-{len(profile.chapter_ids)}.json",
            "style": f"author_style_profile.sample-{len(profile.chapter_ids)}.json",
        },
    })
    report = SkillPackageQualityReport(
        author_id, work_id, len(profile.chapter_ids), len(library.templates), len(profile.constraints), generated, issues,
    )
    report_path = destination / "skill_quality.json"
    write_json(report_path, report.to_dict())
    if not report.passed:
        raise ValueError("compiled skill did not pass package quality checks")
    return destination, report_path
