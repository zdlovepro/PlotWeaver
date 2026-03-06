import re
from typing import Optional

from ..models.novel import Novel, Chapter


# Patterns for Chinese and English chapter headings
_CHAPTER_PATTERNS = [
    re.compile(r"^第[零一二三四五六七八九十百千\d]+[章节回集卷篇][\s　]*(.*)$", re.MULTILINE),
    re.compile(r"^Chapter\s+\d+[\s:：]*(.*)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^CHAPTER\s+\d+[\s:：]*(.*)$", re.MULTILINE),
    re.compile(r"^\d+[\s\.、。]+(.{0,50})$", re.MULTILINE),
]

_CHAPTER_SPLIT_PATTERN = re.compile(
    r"(^第[零一二三四五六七八九十百千\d]+[章节回集卷篇][\s　]*.*)|(^Chapter\s+\d+[\s:：].*)|(^CHAPTER\s+\d+[\s:：].*)",
    re.MULTILINE | re.IGNORECASE,
)


def split_chapters(novel: Novel) -> list[Chapter]:
    """Split novel raw_text into chapters using common chapter heading patterns."""
    text = novel.raw_text

    # Find all chapter heading positions
    matches = list(_CHAPTER_SPLIT_PATTERN.finditer(text))

    if not matches:
        # No chapter headings found – treat entire text as one chapter
        return [
            Chapter(
                id=f"{novel.id}_ch_1",
                novel_id=novel.id,
                number=1,
                title="正文",
                content=text.strip(),
            )
        ]

    chapters: list[Chapter] = []

    # Text before the first chapter heading becomes a prologue (if non-empty)
    prologue = text[: matches[0].start()].strip()
    if prologue:
        chapters.append(
            Chapter(
                id=f"{novel.id}_ch_0",
                novel_id=novel.id,
                number=0,
                title="序章",
                content=prologue,
            )
        )

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        heading = match.group(0).strip()
        content = text[start:end].strip()
        chapters.append(
            Chapter(
                id=f"{novel.id}_ch_{idx + 1}",
                novel_id=novel.id,
                number=idx + 1,
                title=heading,
                content=content,
            )
        )

    return chapters
