import uuid
from typing import Optional

from ..llm.base import BaseLLMClient
from ..llm.prompt_manager import PromptManager
from ..llm.response_parser import parse_plot_units
from ..models.novel import Chapter
from ..models.plot_unit import PlotUnit


class PlotExtractor:
    """Extracts plot units from novel chapters using the LLM 'extract_plots' template."""

    def __init__(self, llm_client: BaseLLMClient, prompt_manager: Optional[PromptManager] = None):
        self.llm = llm_client
        self.prompts = prompt_manager or PromptManager()

    def extract_from_chapter(self, chapter: Chapter) -> list[PlotUnit]:
        """Extract plot units from a single chapter."""
        system = self.prompts.get_system("extract_plots")
        user = self.prompts.get_user("extract_plots", chapter_text=chapter.content)

        from ..llm.base import Message
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]
        response = self.llm.chat(messages)
        raw_units = parse_plot_units(response.content)

        plot_units = []
        for idx, unit_data in enumerate(raw_units):
            plot_unit = PlotUnit(
                id=str(uuid.uuid4()),
                novel_id=chapter.novel_id,
                chapter_id=chapter.id,
                sequence=idx,
                content=unit_data.get("content", ""),
                characters=unit_data.get("characters", []),
                location=unit_data.get("location", ""),
                time_point=unit_data.get("time_point", ""),
                event_type=unit_data.get("event_type", ""),
            )
            plot_units.append(plot_unit)
        return plot_units

    def extract_from_novel(self, chapters: list[Chapter]) -> list[PlotUnit]:
        """Extract plot units from all chapters of a novel."""
        all_units: list[PlotUnit] = []
        for chapter in chapters:
            if not chapter.content.strip():
                continue
            units = self.extract_from_chapter(chapter)
            all_units.extend(units)
        return all_units
