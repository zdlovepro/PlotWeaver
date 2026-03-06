from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlotUnit:
    id: str
    novel_id: str
    chapter_id: str
    sequence: int
    content: str
    characters: list[str] = field(default_factory=list)
    location: str = ""
    time_point: str = ""
    event_type: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "novel_id": self.novel_id,
            "chapter_id": self.chapter_id,
            "sequence": self.sequence,
            "content": self.content,
            "characters": self.characters,
            "location": self.location,
            "time_point": self.time_point,
            "event_type": self.event_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlotUnit":
        return cls(
            id=data["id"],
            novel_id=data["novel_id"],
            chapter_id=data["chapter_id"],
            sequence=data["sequence"],
            content=data["content"],
            characters=data.get("characters", []),
            location=data.get("location", ""),
            time_point=data.get("time_point", ""),
            event_type=data.get("event_type", ""),
        )


@dataclass
class AnnotatedPlotUnit:
    plot_unit: PlotUnit
    narrative_function: str = ""
    emotion_tone: str = ""
    tension_level: float = 0.0
    conflict_type: str = ""
    causal_summary: str = ""
    themes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.plot_unit.to_dict()
        d.update(
            {
                "narrative_function": self.narrative_function,
                "emotion_tone": self.emotion_tone,
                "tension_level": self.tension_level,
                "conflict_type": self.conflict_type,
                "causal_summary": self.causal_summary,
                "themes": self.themes,
            }
        )
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "AnnotatedPlotUnit":
        plot_unit = PlotUnit.from_dict(data)
        return cls(
            plot_unit=plot_unit,
            narrative_function=data.get("narrative_function", ""),
            emotion_tone=data.get("emotion_tone", ""),
            tension_level=float(data.get("tension_level", 0.0)),
            conflict_type=data.get("conflict_type", ""),
            causal_summary=data.get("causal_summary", ""),
            themes=data.get("themes", []),
        )
