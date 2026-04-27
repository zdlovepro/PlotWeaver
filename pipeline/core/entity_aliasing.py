from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


ROLE_KEYWORDS = (
    ("主角", "主角"),
    ("师尊", "引路人"),
    ("师父", "引路人"),
    ("师兄", "同门"),
    ("师姐", "同门"),
    ("师弟", "同门"),
    ("师妹", "同门"),
    ("同门", "同门"),
    ("长老", "前辈"),
    ("前辈", "前辈"),
    ("朋友", "同伴"),
    ("盟友", "同伴"),
    ("同伴", "同伴"),
    ("红颜", "同伴"),
    ("敌人", "对手"),
    ("宿敌", "对手"),
    ("对手", "对手"),
    ("仇人", "对手"),
    ("路人", "旁人"),
    ("陌生", "旁人"),
)

LOCATION_KEYWORDS = (
    ("宗", "宗门"),
    ("门", "宗门"),
    ("城", "城池"),
    ("镇", "城镇"),
    ("坊", "坊市"),
    ("市", "坊市"),
    ("山", "山地"),
    ("谷", "山谷"),
    ("海", "海域"),
    ("湖", "水域"),
    ("河", "水域"),
    ("秘境", "秘境"),
    ("遗迹", "秘境"),
    ("禁地", "禁地"),
    ("洞府", "洞府"),
)

SERIALS = "甲乙丙丁戊己庚辛壬癸"
GENERIC_ROLE_PREFIXES = ("主角", "引路人", "同门", "前辈", "同伴", "对手", "旁人", "角色")
GENERIC_LOCATION_PREFIXES = ("宗门", "城池", "城镇", "坊市", "山地", "山谷", "海域", "水域", "秘境", "禁地", "洞府", "地界")


def split_character_and_relation(raw_character: Any) -> Tuple[str, str]:
    text = str(raw_character or "").strip()
    if not text:
        return "", ""
    match = re.match(r"^(.*?)[(（]([^()（）]+)[)）]$", text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text, ""


def normalize_relation_label(relation: str) -> str:
    text = str(relation or "").strip()
    if not text:
        return "角色"
    for keyword, label in ROLE_KEYWORDS:
        if keyword in text:
            return label
    return "角色"


def infer_location_kind(location: str) -> str:
    text = str(location or "").strip()
    if not text:
        return "地界"
    for keyword, kind in LOCATION_KEYWORDS:
        if keyword in text:
            return kind
    return "地界"


def _serial_suffix(index: int) -> str:
    if index <= 0:
        return "甲"
    if index <= len(SERIALS):
        return SERIALS[index - 1]
    return str(index)


@dataclass
class NovelAliasRegistry:
    novel_name: str = ""
    character_aliases: Dict[str, str] = field(default_factory=dict)
    character_keys: Dict[str, str] = field(default_factory=dict)
    location_aliases: Dict[str, str] = field(default_factory=dict)
    role_counters: Dict[str, int] = field(default_factory=dict)
    location_counters: Dict[str, int] = field(default_factory=dict)

    def _novel_prefix(self) -> str:
        base = str(self.novel_name or "novel").strip()
        base = base.rsplit(".", 1)[0]
        base = re.sub(r"\s+", "", base)
        return base or "novel"

    def alias_character(self, raw_character: Any) -> str:
        raw_name, relation = split_character_and_relation(raw_character)
        if not raw_name:
            return ""
        if raw_name == "主角" or any(raw_name.startswith(prefix) for prefix in GENERIC_ROLE_PREFIXES[1:]):
            self.character_aliases.setdefault(raw_name, raw_name)
            return raw_name
        if raw_name in self.character_aliases:
            return self.character_aliases[raw_name]

        role_label = normalize_relation_label(relation)
        if role_label == "主角":
            alias = "主角"
        else:
            self.role_counters[role_label] = self.role_counters.get(role_label, 0) + 1
            alias = f"{role_label}{_serial_suffix(self.role_counters[role_label])}"
        self.character_aliases[raw_name] = alias
        return alias

    def alias_characters(self, raw_characters: Any, limit: int = 6) -> List[str]:
        if not isinstance(raw_characters, list):
            return []
        aliases: List[str] = []
        for raw_character in raw_characters:
            alias = self.alias_character(raw_character)
            if alias and alias not in aliases:
                aliases.append(alias)
            if len(aliases) >= limit:
                break
        return aliases

    def character_key(self, raw_character: Any) -> str:
        raw_name, _ = split_character_and_relation(raw_character)
        if not raw_name:
            return ""
        if raw_name in self.character_keys:
            return self.character_keys[raw_name]
        key = f"{self._novel_prefix()}_char_{len(self.character_keys) + 1:03d}"
        self.character_keys[raw_name] = key
        return key

    def character_keys_for(self, raw_characters: Any, limit: int = 6) -> List[str]:
        if not isinstance(raw_characters, list):
            return []
        keys: List[str] = []
        for raw_character in raw_characters:
            key = self.character_key(raw_character)
            if key and key not in keys:
                keys.append(key)
            if len(keys) >= limit:
                break
        return keys

    def alias_location(self, raw_location: Any) -> str:
        location = str(raw_location or "").strip()
        if not location:
            return ""
        if any(location.startswith(prefix) for prefix in GENERIC_LOCATION_PREFIXES):
            self.location_aliases.setdefault(location, location)
            return location
        if location in self.location_aliases:
            return self.location_aliases[location]
        kind = infer_location_kind(location)
        self.location_counters[kind] = self.location_counters.get(kind, 0) + 1
        alias = f"{kind}{_serial_suffix(self.location_counters[kind])}"
        self.location_aliases[location] = alias
        return alias


def sanitize_text_entities(
    text: Any,
    raw_characters: Any,
    raw_location: Any,
    alias_registry: NovelAliasRegistry,
) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    if isinstance(raw_characters, list):
        pairs: List[Tuple[str, str]] = []
        for raw_character in raw_characters:
            raw_name, _ = split_character_and_relation(raw_character)
            if not raw_name:
                continue
            alias = alias_registry.alias_character(raw_character)
            if alias:
                pairs.append((raw_name, alias))
        for raw_name, alias in sorted(pairs, key=lambda item: len(item[0]), reverse=True):
            cleaned = cleaned.replace(raw_name, alias)

    raw_location_text = str(raw_location or "").strip()
    if raw_location_text:
        cleaned = cleaned.replace(raw_location_text, alias_registry.alias_location(raw_location_text))

    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[，,]{2,}", "，", cleaned)
    return cleaned.strip(" ，。；;、")


def sanitize_text_list(
    values: Any,
    raw_characters: Any,
    raw_location: Any,
    alias_registry: NovelAliasRegistry,
    limit: int = 8,
) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned_values: List[str] = []
    for value in values:
        cleaned = sanitize_text_entities(value, raw_characters, raw_location, alias_registry)
        if cleaned and cleaned not in cleaned_values:
            cleaned_values.append(cleaned)
        if len(cleaned_values) >= limit:
            break
    return cleaned_values
