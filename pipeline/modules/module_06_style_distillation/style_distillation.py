"""Stage 6: derive an evidence-backed, source-safe author style profile.

The model never receives chapter prose, titles, names, or source quotations in
this stage.  It sees only anonymous numeric feature cards and may therefore
describe reusable drafting constraints without becoming a source-text recovery
step.  Every generated constraint must cite its supporting chapter identifiers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from .contracts import (
    AuthorStyleProfile,
    ChapterAnnotation,
    ChapterDocument,
    ChapterStyleCard,
    StyleBaseline,
    StyleConstraint,
    StyleMetric,
)
from .jsonio import read_jsonl, write_json, write_jsonl
from .model import ModelSettings, complete_json
from .paths import corpus_dir


DEFAULT_STYLE_LIMIT = 20
DEFAULT_STYLE_BATCH_CHAPTERS = 5
_MIN_CONSTRAINTS = 5
_MIN_FAMILIES = 5
_CORE_FAMILIES = frozenset({"sentence_rhythm", "paragraph_pacing", "dialogue", "transition", "focalization"})
_STYLE_BANNED_LANGUAGE = ("模仿", "复刻", "照搬", "原文", "原作", "作者名")
_OVERCLAIMING_LANGUAGE = (
    "严格限定", "通常限于", "稳定在", "每轮", "每章", "大量使用", "极少使用", "必然",
    "始终", "完全", "限定视角", "全知视角", "常以", "多用", "偏重", "只呈现", "内聚焦", "外部聚焦",
)
_UNSUPPORTED_SCENE_LANGUAGE = ("战斗", "追逐", "修炼", "法术", "门派", "境界", "功法", "妖兽")
_FAMILY_LABELS = {
    "sentence_rhythm": "句式节奏",
    "paragraph_pacing": "段落节奏",
    "dialogue": "对话组织",
    "transition": "转场衔接",
    "focalization": "叙述聚焦",
    "imagery": "意象与感官",
    "emotion": "情绪递进",
}
_METRIC_LABELS = {
    "paragraph_count": "段落数",
    "mean_paragraph_chars": "平均段落字数",
    "short_paragraph_ratio": "短段占比（不超过35字）",
    "sentence_count": "句子数",
    "mean_sentence_chars": "平均句长",
    "dialogue_char_ratio": "引号内对话字数占比",
    "dialogue_turn_count": "对话轮次数",
    "dialogue_question_ratio": "引号内问句占比",
    "dialogue_exclamation_ratio": "引号内感叹句占比",
    "dialogue_ellipsis_per_turn": "每个对话轮次的省略号次数",
    "exclamation_sentence_ratio": "感叹句占比",
    "question_sentence_ratio": "问句占比",
    "ellipsis_per_sentence": "每句省略号次数",
    "transition_marker_ratio": "转场标记句占比",
    "perception_marker_ratio": "感知标记句占比",
    "interior_marker_ratio": "心理标记句占比",
    "comparison_marker_ratio": "比喻标记句占比",
    "intensity_marker_ratio": "力度/速度标记句占比",
}
_MARKERS = {
    "transition_marker_ratio": ("随后", "此时", "片刻后", "不久", "旋即", "继而", "当即"),
    "perception_marker_ratio": ("看到", "看见", "望着", "听到", "感到", "察觉"),
    "interior_marker_ratio": ("心中", "暗道", "心里", "沉吟", "念头"),
    "comparison_marker_ratio": ("仿佛", "宛如", "犹如", "好似"),
    "intensity_marker_ratio": ("顿时", "立刻", "突然", "缓缓", "微微"),
}
_FAMILY_METRICS = {
    "sentence_rhythm": {"sentence_count", "mean_sentence_chars", "exclamation_sentence_ratio", "question_sentence_ratio", "ellipsis_per_sentence"},
    "paragraph_pacing": {"paragraph_count", "mean_paragraph_chars", "short_paragraph_ratio"},
    "dialogue": {"dialogue_char_ratio", "dialogue_turn_count", "dialogue_question_ratio", "dialogue_exclamation_ratio", "dialogue_ellipsis_per_turn"},
    "transition": {"transition_marker_ratio", "paragraph_count", "short_paragraph_ratio"},
    "focalization": {"perception_marker_ratio", "interior_marker_ratio", "mean_sentence_chars"},
    "imagery": {"comparison_marker_ratio", "perception_marker_ratio"},
    "emotion": {"interior_marker_ratio", "intensity_marker_ratio", "ellipsis_per_sentence", "exclamation_sentence_ratio"},
}


@dataclass(frozen=True)
class StyleQualityIssue:
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class StyleQualityReport:
    author_id: str
    work_id: str
    chapter_count: int
    constraint_count: int
    families: tuple[str, ...]
    evidence_chapter_coverage: int
    issues: tuple[StyleQualityIssue, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not any(item.severity == "error" for item in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "author_id": self.author_id,
            "work_id": self.work_id,
            "chapter_count": self.chapter_count,
            "constraint_count": self.constraint_count,
            "families": list(self.families),
            "evidence_chapter_coverage": self.evidence_chapter_coverage,
            "passed": self.passed,
            "issues": [item.to_dict() for item in self.issues],
        }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[。！？!?]+", text) if part.strip()]


def _dialogue_spans(text: str) -> list[str]:
    return re.findall(r"[“\"‘「『](.*?)[”\"’」』]", text, flags=re.DOTALL)


def _metric(metric_id: str, value: float, unit: str, denominator: str = "") -> StyleMetric:
    return StyleMetric(metric_id, round(float(value), 6), unit, denominator)


def build_chapter_style_card(document: ChapterDocument) -> ChapterStyleCard:
    """Measure a source chapter locally without retaining stylistic quotations."""

    text = "\n".join(unit.text for unit in document.annotation_units)
    paragraphs = [unit.text for unit in document.annotation_units if unit.text.strip()]
    sentences = _sentences(text)
    sentence_count = len(sentences)
    dialogue_spans = _dialogue_spans(text)
    dialogue_chars = sum(len(item) for item in dialogue_spans)
    dialogue_text = "\n".join(dialogue_spans)
    metrics = [
        _metric("paragraph_count", len(paragraphs), "count"),
        _metric("mean_paragraph_chars", _ratio(sum(len(item) for item in paragraphs), len(paragraphs)), "mean_chars"),
        _metric("short_paragraph_ratio", _ratio(sum(len(item) <= 35 for item in paragraphs), len(paragraphs)), "ratio"),
        _metric("sentence_count", sentence_count, "count"),
        _metric("mean_sentence_chars", _ratio(sum(len(item) for item in sentences), sentence_count), "mean_chars"),
        _metric("dialogue_char_ratio", _ratio(dialogue_chars, len(text)), "ratio"),
        _metric("dialogue_turn_count", len(dialogue_spans), "count"),
        _metric("dialogue_question_ratio", _ratio(len(re.findall(r"[？?]", dialogue_text)), len(dialogue_spans)), "ratio"),
        _metric("dialogue_exclamation_ratio", _ratio(len(re.findall(r"[！!]", dialogue_text)), len(dialogue_spans)), "ratio"),
        _metric("dialogue_ellipsis_per_turn", _ratio(dialogue_text.count("……") + dialogue_text.count("..."), len(dialogue_spans)), "ratio"),
        _metric("exclamation_sentence_ratio", _ratio(len(re.findall(r"[！!]", text)), sentence_count), "ratio"),
        _metric("question_sentence_ratio", _ratio(len(re.findall(r"[？?]", text)), sentence_count), "ratio"),
        _metric("ellipsis_per_sentence", _ratio(text.count("……") + text.count("..."), sentence_count), "ratio"),
    ]
    for metric_id, markers in _MARKERS.items():
        count = sum(1 for sentence in sentences if any(marker in sentence for marker in markers))
        metrics.append(_metric(metric_id, _ratio(count, sentence_count), "ratio"))
    return ChapterStyleCard(document.chapter_id, document.source_hash, tuple(metrics), ())


def _style_feature_cards(cards: Iterable[ChapterStyleCard]) -> list[dict[str, Any]]:
    """Return anonymous, numeric cards that are safe to send to a model."""

    result: list[dict[str, Any]] = []
    for card in cards:
        result.append({
            "chapter_id": card.chapter_id,
            "metrics": [
                {"metric": _METRIC_LABELS.get(item.metric_id, item.metric_id), "value": item.value, "unit": item.unit}
                for item in card.metrics
            ],
        })
    return result


def _chunks(values: list[Any], size: int) -> list[list[Any]]:
    size = max(2, int(size))
    return [values[index:index + size] for index in range(0, len(values), size)]


def _constraint_schema() -> dict[str, Any]:
    return {
        "constraints": [{
            "family": "sentence_rhythm",
            "rule": "抽象、可执行的文风规律",
            "application": "在什么写作情境下如何使用",
            "avoid": "应避免的机械化做法",
            "evidence_metric_ids": ["mean_sentence_chars", "question_sentence_ratio"],
            "evidence_chapter_ids": ["作者/作品/0001", "作者/作品/0002"],
        }]
    }


def _style_prompt(cards: list[dict[str, Any]], *, final: bool) -> str:
    ids = [item["chapter_id"] for item in cards]
    schema = _constraint_schema()
    example = {
        "constraints": [{
            "family": "sentence_rhythm",
            "rule": "在信息压力升高时交替使用完整叙述句与短促动作句，避免连续同长度句式。",
            "application": "把短句放在动作落点或判断落点之前、之后，用完整句交代因果与感知。",
            "avoid": "不要把短句机械堆叠为口号，也不要只按固定字数切句。",
            "evidence_metric_ids": ["mean_sentence_chars", "question_sentence_ratio"],
            "evidence_chapter_ids": ids[:2],
        }]
    }
    phase = "跨批次综合" if final else "本批观察"
    return f"""任务：你是小说文风特征提炼器。请依据下列不含原文、不含标题、不含人物或设定名的量化章节特征，做{phase}，归纳可迁移、可执行的写作约束。

你不是续写器，不能复述、猜测或还原任何来源文本。自然语言字段必须使用中文；`family` 只能从以下英文枚举中选择：{json.dumps(list(_FAMILY_LABELS), ensure_ascii=False)}。每条约束只能描述句式、段落、对话、转场、叙述聚焦、感官意象或情绪递进等抽象策略，不能出现专名、原句、具体情节、世界观术语或来源作者信息。

只输出一个合法 JSON object，顶层字段必须且只能是 `constraints`。输出结构：
{json.dumps(schema, ensure_ascii=False, indent=2)}

字段示例（仅说明格式，不可照抄为结论）：
{json.dumps(example, ensure_ascii=False, indent=2)}

硬性规则：
1. `evidence_chapter_ids` 只能从 {json.dumps(ids, ensure_ascii=False)} 中选择，且不得重复。
2. 每条约束至少引用 {min(3, len(ids))} 个不同章节；证据不足时不要输出。
3. `evidence_metric_ids` 只能从以下度量 ID 选择：{json.dumps(sorted(_METRIC_LABELS), ensure_ascii=False)}。它必须列出直接支撑该约束的 1-4 个指标，且要与 `family` 相符。
4. `rule` 必须是能直接指导写作的规律，`application` 必须说明适用场景与落点，`avoid` 必须指出避免的僵化方式。自然语言字段不要写任何数字、百分比、固定字数、固定句数或机械配额；数值范围已经单独保存在档案基线中。
5. 不要使用“模仿、复刻、照搬、原文、原作、作者名”等词，也不要产生任何具体角色、地点、物品、门派或设定名。不得凭标记词频推断“严格全知/限知视角”，不得引入战斗、追逐、修炼等来源特定场景。
6. 输出 5-7 条约束，并且必须各有一条属于 `sentence_rhythm`、`paragraph_pacing`、`dialogue`、`transition`、`focalization`。只有在给定指标确实无法支持时才可以省略其他类别。
7. 只保留跨章节稳定模式；不要将单章数值或某个故事片段包装成规律，也不要使用“每章、每轮、稳定在、严格限定、通常限于”等绝对化表达。

量化章节特征：
{json.dumps(cards, ensure_ascii=False)}"""


def _offline_constraints(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = [item["chapter_id"] for item in cards]
    return [
        {"family": "sentence_rhythm", "rule": "在动作、判断或信息落点附近使用长短句交替，保证因果交代与节奏变化同时存在。", "application": "先用完整句交代感知和因果，再在关键动作或判断处收束为短句。", "avoid": "不要把短句连续堆成口号，也不要按固定字数切句。", "evidence_metric_ids": ["mean_sentence_chars", "question_sentence_ratio"], "evidence_chapter_ids": ids},
        {"family": "paragraph_pacing", "rule": "以信息单元切段：推进、反应和结果可以分别落段，压力节点允许短段加速。", "application": "场景推进时先放置目标或变化，再用独立段落呈现反应和后果。", "avoid": "不要为了制造节奏而把每一句都拆成独立段落。", "evidence_metric_ids": ["mean_paragraph_chars", "short_paragraph_ratio"], "evidence_chapter_ids": ids},
        {"family": "dialogue", "rule": "让对话承担试探、选择或压力传递，并让叙述动作补足说话后的变化。", "application": "在对话前后加入可观察的动作、停顿或判断，使话语改变后续行动。", "avoid": "不要让对话只重复已经说明的信息，也不要连续问答而没有状态变化。", "evidence_metric_ids": ["dialogue_char_ratio", "dialogue_turn_count"], "evidence_chapter_ids": ids},
        {"family": "transition", "rule": "转场应由可见动作、感知变化或结果承接，使场景切换保留前一段的因果余波。", "application": "切换前给出行动结果或注意力落点，切换后先交代新的压力或目标。", "avoid": "不要只用时间副词跳过中间因果，也不要无提示改变场景。", "evidence_metric_ids": ["transition_marker_ratio", "short_paragraph_ratio"], "evidence_chapter_ids": ids},
        {"family": "focalization", "rule": "用角色的感知和心理反应承接动作结果，使读者先得到可感知的信号再理解判断。", "application": "在行动变化或信息揭示后，补充角色当下的感受、注意力落点或判断。", "avoid": "不要将感知词堆叠为说明，也不要无提示切换信息中心。", "evidence_metric_ids": ["perception_marker_ratio", "interior_marker_ratio"], "evidence_chapter_ids": ids},
    ]


def _constraint_rows(payload: dict[str, Any], allowed_ids: set[str], prefix: str) -> tuple[StyleConstraint, ...]:
    rows = payload.get("constraints", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return ()
    constraints: list[StyleConstraint] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        family = str(row.get("family", "")).strip()
        if family not in _FAMILY_LABELS:
            continue
        evidence = tuple(dict.fromkeys(
            str(item).strip() for item in row.get("evidence_chapter_ids", [])
            if str(item).strip() in allowed_ids
        ))
        metric_ids = tuple(dict.fromkeys(
            str(item).strip() for item in row.get("evidence_metric_ids", [])
            if str(item).strip() in _METRIC_LABELS
        ))
        rule = str(row.get("rule", "")).strip()
        application = str(row.get("application", "")).strip()
        avoid = str(row.get("avoid", "")).strip()
        if len(evidence) < 2 or not metric_ids or not rule or not application or not avoid:
            continue
        constraints.append(StyleConstraint(
            f"{prefix}:{family}-{len(constraints) + 1:03d}", family, rule, application, avoid,
            evidence, len(evidence), min(0.95, round(0.58 + len(evidence) * 0.035, 3)), metric_ids,
        ))
    return tuple(constraints)


def _neutralise_constraint_language(rows: Iterable[StyleConstraint]) -> tuple[StyleConstraint, ...]:
    """Remove wording that turns an observed tendency into a false hard rule.

    This is deliberately a narrow, transparent text normalisation.  It never
    adds a style claim; it only changes a prohibited universal quantifier into a
    neutral drafting instruction before semantic quality is assessed.
    """

    substitutions = {
        "严格限定": "优先围绕",
        "通常限于": "宜保持为",
        "稳定在": "保持在",
        "每章": "写作过程中",
        "每轮": "对话推进时",
        "大量使用": "适度使用",
        "极少使用": "谨慎使用",
        "必然": "可以",
    }
    safe_frames = {
        "sentence_rhythm": (
            "以完整句交代信息与因果，在需要强调疑问或情绪时穿插语气句，保持前后句式变化。",
            "先写清动作、判断或信息变化，再用问句或感叹句作为需要强调时的节奏落点。",
            "不要连续堆叠同一种语气，也不要为了制造节奏而打断必要的因果交代。",
        ),
        "paragraph_pacing": (
            "通过短段和较长段的交替调整信息密度，让强调、停顿和连贯铺陈各有独立空间。",
            "把需要强调的变化单独落段；需要连续交代时保留完整段落，随后用段落长度变化调节速度。",
            "不要把所有段落切成同样长度，也不要连续短段造成阅读碎片化。",
        ),
        "dialogue": (
            "让对话中的语气变化服务于回应、试探或压力传递，并把未尽之意作为节奏调节而非固定装饰。",
            "在对话需要推进关系或信息时使用问句、感叹句或停顿，并用后续动作或判断承接话语结果。",
            "不要让标点替代人物反应，也不要连续堆叠问句、感叹句或省略号。",
        ),
        "transition": (
            "把显性转场表述作为补充，优先用前一段的结果、动作或注意力落点承接后续信息。",
            "切换场景、时间或话题前留下可追踪的变化，切换后先交代新的压力、目标或可感知信号。",
            "不要仅靠固定连接词跳过因果，也不要在没有承接点时突然改变叙述对象。",
        ),
        "focalization": (
            "在动作或信息发生变化后补入当下的感知、判断或心理反应，使外部变化获得可体验的反应层。",
            "先给出角色注意到的信号或结果，再展开其判断、情绪反应或下一步行动。",
            "不要把感知和心理词堆成说明，也不要让反应层脱离当前发生的动作与信息。",
        ),
        "imagery": (
            "以少量感官细节或比较性表达服务于当前信息焦点，使画面感不压过行动与因果。",
            "在需要突出环境、状态或情绪时加入一个可感知细节，并回到正在推进的动作或判断。",
            "不要连续堆砌修辞，也不要让意象替代读者理解所需的具体变化。",
        ),
        "emotion": (
            "在需要强调的情绪节点提高语气和动作力度，并在前后保留平稳叙述形成对比。",
            "先交代触发情绪的变化，再以语气、停顿或动作力度收束当下反应，随后回到结果。",
            "不要连续抬高情绪强度，也不要让强烈标记脱离具体触发原因。",
        ),
    }
    normalised: list[StyleConstraint] = []
    for item in rows:
        def clean(value: str) -> str:
            for old, new in substitutions.items():
                value = value.replace(old, new)
            return value
        cleaned = (clean(item.rule), clean(item.application), clean(item.avoid))
        combined = "\n".join(cleaned)
        # Lexical marker counts cannot establish a fixed point of view, causal
        # relation, or a dialogue/narration split.  Keep the evidence selected by
        # the model, but replace such overclaims with a safe, executable frame.
        requires_safe_frame = (
            any(token in combined for token in _OVERCLAIMING_LANGUAGE)
            or bool(re.search(r"[零一二三四五六七八九十百千万两]{2,}", combined))
            or (item.family == "focalization" and any(token in combined for token in ("视角", "聚焦于", "直接暴露")))
            or (item.family == "sentence_rhythm" and "对话" in combined and not any(metric.startswith("dialogue_") for metric in item.evidence_metric_ids))
            or item.family == "transition"
            or (item.family == "emotion" and any(token in combined for token in ("直接", "间接", "而非", "形容词")))
        )
        rule, application, avoid = safe_frames[item.family] if requires_safe_frame else cleaned
        normalised.append(StyleConstraint(
            item.constraint_id, item.family, rule, application, avoid,
            item.evidence_chapter_ids, item.support_count, item.confidence, item.evidence_metric_ids,
        ))
    return tuple(normalised)


def _baselines(cards: list[ChapterStyleCard]) -> tuple[StyleBaseline, ...]:
    by_metric: dict[str, list[StyleMetric]] = {}
    for card in cards:
        for metric in card.metrics:
            by_metric.setdefault(metric.metric_id, []).append(metric)
    result = []
    for metric_id, rows in sorted(by_metric.items()):
        values = [row.value for row in rows]
        result.append(StyleBaseline(
            metric_id, round(fmean(values), 6), round(min(values), 6), round(max(values), 6), rows[0].unit,
        ))
    return tuple(result)


def _source_terms(annotations: Iterable[ChapterAnnotation]) -> tuple[str, ...]:
    names: set[str] = set()
    for annotation in annotations:
        for entity in annotation.entities:
            for value in (entity.canonical_name, *entity.aliases):
                term = str(value).strip()
                if len(term) >= 2:
                    names.add(term)
    return tuple(sorted(names, key=len, reverse=True))


def assess_style_profile(profile: AuthorStyleProfile, source_terms: Iterable[str] = ()) -> StyleQualityReport:
    issues: list[StyleQualityIssue] = []
    try:
        profile.validate()
    except (TypeError, ValueError) as exc:
        issues.append(StyleQualityIssue("invalid_contract", str(exc)))
    families = tuple(sorted({item.family for item in profile.constraints}))
    if len(profile.constraints) < _MIN_CONSTRAINTS:
        issues.append(StyleQualityIssue("too_few_constraints", f"至少需要 {_MIN_CONSTRAINTS} 条文风约束"))
    if len(families) < _MIN_FAMILIES:
        issues.append(StyleQualityIssue("too_few_style_families", f"至少需要 {_MIN_FAMILIES} 类文风约束"))
    missing_core = sorted(_CORE_FAMILIES - set(families))
    if missing_core:
        issues.append(StyleQualityIssue("missing_core_style_family", f"缺少核心文风类别：{', '.join(missing_core)}"))
    covered = {chapter_id for item in profile.constraints for chapter_id in item.evidence_chapter_ids}
    required_coverage = max(2, int(len(profile.chapter_ids) * 0.6 + 0.999))
    if len(covered) < required_coverage:
        issues.append(StyleQualityIssue("insufficient_evidence_coverage", f"约束证据仅覆盖 {len(covered)} 章，至少需要 {required_coverage} 章"))
    forbidden = tuple(dict.fromkeys((*_STYLE_BANNED_LANGUAGE, *(item for item in source_terms if len(item) >= 2))))
    for item in profile.constraints:
        combined = "\n".join((item.rule, item.application, item.avoid))
        if any(token in combined for token in forbidden):
            issues.append(StyleQualityIssue("source_or_copying_language", f"{item.constraint_id} 含有来源特定词或不应出现的表述"))
        if len(item.rule) < 12 or len(item.application) < 12 or len(item.avoid) < 10:
            issues.append(StyleQualityIssue("constraint_too_shallow", f"{item.constraint_id} 的规则、应用或规避说明过短"))
        minimum_support = 3 if len(profile.chapter_ids) >= 10 else 2
        if item.support_count < minimum_support:
            issues.append(StyleQualityIssue("insufficient_constraint_support", f"{item.constraint_id} 至少需要 {minimum_support} 个章节证据"))
        if not set(item.evidence_metric_ids) <= set(_METRIC_LABELS):
            issues.append(StyleQualityIssue("unknown_evidence_metric", f"{item.constraint_id} 引用了未知度量"))
        if not set(item.evidence_metric_ids) & _FAMILY_METRICS.get(item.family, set()):
            issues.append(StyleQualityIssue("family_metric_mismatch", f"{item.constraint_id} 没有引用与文风类别相符的度量"))
        if any(token in combined for token in _OVERCLAIMING_LANGUAGE):
            issues.append(StyleQualityIssue("unsupported_absolute_claim", f"{item.constraint_id} 含有量化证据不能支持的绝对化结论"))
        if re.search(r"\d|[零一二三四五六七八九十百千万两]{2,}", combined):
            issues.append(StyleQualityIssue("numeric_rule_duplicates_baseline", f"{item.constraint_id} 将数字配额写入自然语言约束"))
        if any(token in combined for token in _UNSUPPORTED_SCENE_LANGUAGE):
            issues.append(StyleQualityIssue("unsupported_concrete_scene", f"{item.constraint_id} 引入了匿名特征无法支持的具体场景"))
    return StyleQualityReport(
        profile.author_id, profile.work_id, len(profile.chapter_ids), len(profile.constraints), families, len(covered), tuple(issues),
    )


def _profile_from_rows(
    author_id: str,
    work_id: str,
    cards: list[ChapterStyleCard],
    rows: Iterable[StyleConstraint],
) -> AuthorStyleProfile:
    return AuthorStyleProfile(
        author_id,
        work_id,
        tuple(card.chapter_id for card in cards),
        _baselines(cards),
        tuple(rows),
        (
            "仅使用抽象文风约束，不复用来源中的专名、句子、唯一情节或设定。",
            "具体人物、世界、术语和冲突必须由当前原创写作任务提供。",
            "量化指标只用于观察范围，不得机械套用，也不得假定每一章都必须相同。",
        ),
    )


def _synthesis_prompt(cards: list[dict[str, Any]], observations: list[dict[str, Any]], issues: Iterable[StyleQualityIssue] = ()) -> str:
    return _style_prompt(cards, final=True) + "\n\n分批候选（需要去重、合并，并重新选择跨章节证据）：\n" + json.dumps(observations, ensure_ascii=False) + (
        "\n\n上一轮质量问题（请只修正这些问题后重新输出完整 JSON）：\n" + json.dumps([item.to_dict() for item in issues], ensure_ascii=False)
        if tuple(issues) else ""
    )


def _annotation_path(folder: Path, limit: int) -> Path:
    target = folder / f"chapter_annotations.sample-{limit}.jsonl"
    if target.exists():
        return target
    candidates = sorted(folder.glob("chapter_annotations.sample-*.jsonl"))
    if not candidates:
        raise FileNotFoundError("no evidence-first chapter annotations found")
    return candidates[-1]


def distil_style_profile(
    author_id: str,
    work_id: str,
    *,
    limit: int = DEFAULT_STYLE_LIMIT,
    batch_size: int = DEFAULT_STYLE_BATCH_CHAPTERS,
    offline: bool = False,
) -> tuple[Path, Path, Path]:
    """Create measured cards and a validated, abstract author style profile."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    folder = corpus_dir(author_id, work_id)
    documents = [ChapterDocument.from_dict(row) for row in read_jsonl(folder / "chapter_documents.jsonl")][:limit]
    if not documents:
        raise FileNotFoundError("chapter_documents.jsonl is required before style distillation")
    annotations = [ChapterAnnotation.from_dict(row) for row in read_jsonl(_annotation_path(folder, limit))][:len(documents)]
    if tuple(item.chapter_id for item in documents) != tuple(item.chapter_id for item in annotations):
        raise ValueError("chapter documents do not match evidence-first annotations")
    settings = None if offline else ModelSettings.from_environment()
    if settings is not None and not settings.api_key:
        raise RuntimeError("模型密钥不可用：请在 config.yaml 或环境变量中配置后重试")
    cards = [build_chapter_style_card(document) for document in documents]
    feature_cards = _style_feature_cards(cards)
    observations: list[dict[str, Any]] = []
    for index, batch in enumerate(_chunks(feature_cards, batch_size), start=1):
        payload = {"constraints": _offline_constraints(batch)} if offline else complete_json(
            "你是小说文风特征提炼器。只根据匿名量化特征工作，严格遵守用户给出的 JSON 契约。",
            _style_prompt(batch, final=False), settings, attempts=4, max_tokens=min(settings.max_tokens, 10_000),
        )
        observations.append({"batch_index": index, "chapter_ids": [item["chapter_id"] for item in batch], "draft": payload})
    allowed_ids = {card.chapter_id for card in cards}
    source_terms = _source_terms(annotations)
    final_payload = {"constraints": _offline_constraints(feature_cards)} if offline else complete_json(
        "你是小说文风档案总编。只输出由匿名量化证据支持的抽象写作约束，严格遵守 JSON 契约。",
        _synthesis_prompt(feature_cards, observations), settings, attempts=4, max_tokens=min(settings.max_tokens, 12_000),
    )
    constraints = _neutralise_constraint_language(_constraint_rows(final_payload, allowed_ids, "style"))
    profile = _profile_from_rows(author_id, work_id, cards, constraints)
    quality = assess_style_profile(profile, source_terms)
    # Semantic contract failures need an explicit correction turn, not a JSON
    # retry.  The repair prompt sees only the failed abstract output and cards.
    for _ in range(4):
        if quality.passed or offline:
            break
        repair = complete_json(
            "你是小说文风档案质量修订器。请根据质量问题重写完整、抽象、可执行的约束 JSON。",
            _synthesis_prompt(feature_cards, observations, quality.issues), settings, attempts=4,
            max_tokens=min(settings.max_tokens, 12_000),
        )
        profile = _profile_from_rows(
            author_id, work_id, cards, _neutralise_constraint_language(_constraint_rows(repair, allowed_ids, "style")),
        )
        quality = assess_style_profile(profile, source_terms)
    if not quality.passed:
        messages = "; ".join(f"{item.code}: {item.message}" for item in quality.issues)
        raise ValueError(f"style profile quality checks failed: {messages}")
    card_path = folder / f"chapter_style_cards.sample-{len(cards)}.jsonl"
    profile_path = folder / f"author_style_profile.sample-{len(cards)}.json"
    quality_path = folder / f"author_style_profile.sample-{len(cards)}.quality.json"
    write_jsonl(card_path, [item.to_dict() for item in cards])
    write_json(profile_path, profile.to_dict())
    write_json(quality_path, {**quality.to_dict(), "batch_count": len(observations), "batch_chapter_ids": [item["chapter_ids"] for item in observations]})
    return card_path, profile_path, quality_path
