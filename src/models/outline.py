from dataclasses import dataclass, field


@dataclass
class ChapterOutline:
    number: int
    title: str
    summary: str
    key_events: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    emotion_tone: str = ""
    tension_level: float = 0.0

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "summary": self.summary,
            "key_events": self.key_events,
            "characters": self.characters,
            "emotion_tone": self.emotion_tone,
            "tension_level": self.tension_level,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChapterOutline":
        return cls(
            number=data["number"],
            title=data["title"],
            summary=data["summary"],
            key_events=data.get("key_events", []),
            characters=data.get("characters", []),
            emotion_tone=data.get("emotion_tone", ""),
            tension_level=float(data.get("tension_level", 0.0)),
        )


@dataclass
class Outline:
    title: str
    premise: str
    theme: str
    chapters: list[ChapterOutline] = field(default_factory=list)
    source_novels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "premise": self.premise,
            "theme": self.theme,
            "chapters": [c.to_dict() for c in self.chapters],
            "source_novels": self.source_novels,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Outline":
        chapters = [ChapterOutline.from_dict(c) for c in data.get("chapters", [])]
        return cls(
            title=data["title"],
            premise=data["premise"],
            theme=data["theme"],
            chapters=chapters,
            source_novels=data.get("source_novels", []),
        )
