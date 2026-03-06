from dataclasses import dataclass, field


@dataclass
class Chapter:
    id: str
    novel_id: str
    number: int
    title: str
    content: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "novel_id": self.novel_id,
            "number": self.number,
            "title": self.title,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chapter":
        return cls(
            id=data["id"],
            novel_id=data["novel_id"],
            number=data["number"],
            title=data["title"],
            content=data["content"],
        )


@dataclass
class Novel:
    id: str
    title: str
    file_path: str
    chapters: list[Chapter] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "file_path": self.file_path,
            "chapters": [c.to_dict() for c in self.chapters],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Novel":
        chapters = [Chapter.from_dict(c) for c in data.get("chapters", [])]
        return cls(
            id=data["id"],
            title=data["title"],
            file_path=data.get("file_path", ""),
            chapters=chapters,
        )
