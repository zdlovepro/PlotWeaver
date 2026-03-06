import json
from typing import Optional

from ..llm.base import BaseLLMClient, Message
from ..llm.prompt_manager import PromptManager
from ..llm.response_parser import parse_outline
from ..models.outline import Outline, ChapterOutline
from ..models.plot_unit import AnnotatedPlotUnit
from ..models.character import CharacterMapping


class OutlineGenerator:
    """Generates a new novel outline using LLM with the 'generate_outline' template."""

    def __init__(self, llm_client: BaseLLMClient, prompt_manager: Optional[PromptManager] = None):
        self.llm = llm_client
        self.prompts = prompt_manager or PromptManager()

    def generate(
        self,
        remixed_units: list[AnnotatedPlotUnit],
        character_mappings: list[CharacterMapping],
        emotion_curve: Optional[dict] = None,
        tension_curve: Optional[dict] = None,
    ) -> Outline:
        """Generate a complete novel outline from remixed plot units."""
        # Prepare prompt inputs
        plots_summary = self._format_plots(remixed_units)
        char_summary = self._format_character_mappings(character_mappings)
        emotion_req = self._format_emotion_requirement(emotion_curve)
        tension_req = self._format_tension_requirement(tension_curve)

        system = self.prompts.get_system("generate_outline")
        user = self.prompts.get_user(
            "generate_outline",
            remixed_plots=plots_summary,
            character_mapping=char_summary,
            emotion_curve_requirement=emotion_req,
            tension_curve_requirement=tension_req,
        )

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]
        response = self.llm.chat(messages)
        outline_data = parse_outline(response.content)

        return self._build_outline(outline_data)

    def _format_plots(self, units: list[AnnotatedPlotUnit]) -> str:
        lines = []
        for i, u in enumerate(units, 1):
            lines.append(
                f"{i}. [{u.narrative_function}|{u.emotion_tone}|张力{u.tension_level:.1f}] {u.plot_unit.content[:100]}"
            )
        return "\n".join(lines)

    def _format_character_mappings(self, mappings: list[CharacterMapping]) -> str:
        if not mappings:
            return "无明确人物映射"
        lines = []
        for m in mappings:
            lines.append(
                f"- {m.source_character_id} → {m.target_character_id} (相似度: {m.similarity_score:.2f})"
            )
        return "\n".join(lines)

    def _format_emotion_requirement(self, emotion_curve: Optional[dict]) -> str:
        if not emotion_curve:
            return "情感曲线：从平静开始，逐步积累张力，在高潮处达到情感顶点，结局趋于平静"
        turning_points = emotion_curve.get("turning_points", [])
        labels = emotion_curve.get("emotion_labels", [])
        return f"情感曲线共有 {len(turning_points)} 个转折点，主要情感序列：{' → '.join(labels[:5])}"

    def _format_tension_requirement(self, tension_curve: Optional[dict]) -> str:
        if not tension_curve:
            return "张力曲线：整体呈上升趋势，在故事中后期出现高潮，结局张力适当降低"
        peaks = tension_curve.get("peaks", [])
        raw = tension_curve.get("raw", [])
        max_t = max(raw) if raw else 1.0
        return f"张力曲线最高值 {max_t:.2f}，共有 {len(peaks)} 个张力高峰"

    def _build_outline(self, data: dict) -> Outline:
        chapters = []
        for ch_data in data.get("chapters", []):
            chapters.append(
                ChapterOutline(
                    number=int(ch_data.get("number", 0)),
                    title=ch_data.get("title", ""),
                    summary=ch_data.get("summary", ""),
                    key_events=ch_data.get("key_events", []),
                    characters=ch_data.get("characters", []),
                    emotion_tone=ch_data.get("emotion_tone", ""),
                    tension_level=float(ch_data.get("tension_level", 0.5)),
                )
            )
        return Outline(
            title=data.get("title", "新小说"),
            premise=data.get("premise", ""),
            theme=data.get("theme", ""),
            chapters=chapters,
            source_novels=data.get("source_novels", []),
        )
