from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Character:
    id: str
    novel_id: str
    name: str
    role: str = ""
    traits: list[str] = field(default_factory=list)
    relationships: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "novel_id": self.novel_id,
            "name": self.name,
            "role": self.role,
            "traits": self.traits,
            "relationships": self.relationships,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Character":
        return cls(
            id=data["id"],
            novel_id=data["novel_id"],
            name=data["name"],
            role=data.get("role", ""),
            traits=data.get("traits", []),
            relationships=data.get("relationships", {}),
            description=data.get("description", ""),
        )


@dataclass
class CharacterMapping:
    source_character_id: str
    target_character_id: str
    similarity_score: float
    source_novel_id: str
    target_novel_id: str

    def to_dict(self) -> dict:
        return {
            "source_character_id": self.source_character_id,
            "target_character_id": self.target_character_id,
            "similarity_score": self.similarity_score,
            "source_novel_id": self.source_novel_id,
            "target_novel_id": self.target_novel_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CharacterMapping":
        return cls(
            source_character_id=data["source_character_id"],
            target_character_id=data["target_character_id"],
            similarity_score=float(data["similarity_score"]),
            source_novel_id=data["source_novel_id"],
            target_novel_id=data["target_novel_id"],
        )
