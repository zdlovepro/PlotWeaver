from typing import Optional

from ..llm.base import BaseLLMClient, Message
from ..llm.prompt_manager import PromptManager
from ..llm.response_parser import parse_annotation
from ..models.plot_unit import PlotUnit, AnnotatedPlotUnit


# Mapping for narrative function statistics
_NARRATIVE_FUNCTIONS = [
    "开端", "发展", "转折", "高潮", "结局", "铺垫", "悬念", "回忆"
]


class NarrativeAnnotator:
    """Annotates plot units with narrative function, emotion, tension, conflict, causality using LLM."""

    def __init__(self, llm_client: BaseLLMClient, prompt_manager: Optional[PromptManager] = None):
        self.llm = llm_client
        self.prompts = prompt_manager or PromptManager()

    def annotate(self, plot_unit: PlotUnit) -> AnnotatedPlotUnit:
        """Annotate a single plot unit."""
        system = self.prompts.get_system("annotate_plot")
        user = self.prompts.get_user(
            "annotate_plot",
            plot_content=plot_unit.content,
            characters=", ".join(plot_unit.characters) if plot_unit.characters else "未知",
            event_type=plot_unit.event_type or "其他",
        )
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]
        response = self.llm.chat(messages)
        annotation = parse_annotation(response.content)

        return AnnotatedPlotUnit(
            plot_unit=plot_unit,
            narrative_function=annotation.get("narrative_function", ""),
            emotion_tone=annotation.get("emotion_tone", ""),
            tension_level=float(annotation.get("tension_level", 0.0)),
            conflict_type=annotation.get("conflict_type", ""),
            causal_summary=annotation.get("causal_summary", ""),
            themes=annotation.get("themes", []),
        )

    def annotate_all(self, plot_units: list[PlotUnit]) -> list[AnnotatedPlotUnit]:
        """Annotate all plot units."""
        return [self.annotate(unit) for unit in plot_units]

    @staticmethod
    def compute_narrative_stats(annotated_units: list[AnnotatedPlotUnit]) -> dict[str, int]:
        """Compute counts of each narrative function across all annotated units."""
        stats: dict[str, int] = {fn: 0 for fn in _NARRATIVE_FUNCTIONS}
        stats["其他"] = 0
        for unit in annotated_units:
            fn = unit.narrative_function
            if fn in stats:
                stats[fn] += 1
            else:
                stats["其他"] += 1
        return stats
