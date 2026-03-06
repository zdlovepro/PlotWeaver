from ..models.outline import Outline, ChapterOutline


class OutlineFormatter:
    """Formats a novel Outline object into final Markdown output."""

    def to_markdown(self, outline: Outline) -> str:
        """Convert an Outline to a Markdown string."""
        lines = []

        # Title
        lines.append(f"# {outline.title}")
        lines.append("")

        # Meta info
        if outline.premise:
            lines.append("## 故事前提")
            lines.append("")
            lines.append(outline.premise)
            lines.append("")

        if outline.theme:
            lines.append(f"**核心主题**：{outline.theme}")
            lines.append("")

        if outline.source_novels:
            lines.append(f"**素材来源**：{', '.join(outline.source_novels)}")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Chapters
        lines.append("## 章节大纲")
        lines.append("")
        for ch in outline.chapters:
            lines.append(f"### 第{ch.number}章　{ch.title}")
            lines.append("")
            lines.append(f"**情感基调**：{ch.emotion_tone}　**张力值**：{ch.tension_level:.1f}")
            lines.append("")
            lines.append(ch.summary)
            lines.append("")

            if ch.key_events:
                lines.append("**关键事件**：")
                for event in ch.key_events:
                    lines.append(f"- {event}")
                lines.append("")

            if ch.characters:
                lines.append(f"**出场人物**：{', '.join(ch.characters)}")
                lines.append("")

        return "\n".join(lines)

    def save(self, outline: Outline, output_path: str) -> None:
        """Save the formatted outline to a Markdown file."""
        import os
        dirname = os.path.dirname(output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        content = self.to_markdown(outline)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
