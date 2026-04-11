import re
from typing import List, Dict


class OutlinePreprocessor:
    """大纲预处理器"""

    @staticmethod
    def clean_outline(text: str) -> str:
        """清理和标准化大纲文本"""
        if not text:
            return ""

        # 移除多余的空行和空格
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # 标准化标点
        text = re.sub(r'，', ',', text)
        text = re.sub(r'。', '.', text)
        text = re.sub(r'！', '!', text)
        text = re.sub(r'？', '?', text)

        return text.strip()

    @staticmethod
    def split_into_sections(outline: str) -> List[Dict[str, str]]:
        """将大纲分割成章节"""
        sections = []
        lines = outline.split('\n')

        current_section = {"title": "", "content": ""}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测章节标题（包含数字或特定关键词）
            if (re.match(r'^(第[零一二三四五六七八九十百千]+章|Chapter\s+\d+|【.*】|《.*》)', line) or
                    (len(line) < 50 and any(keyword in line for keyword in ['开头', '发展', '高潮', '结局']))):

                # 保存前一个章节
                if current_section["content"]:
                    sections.append(current_section.copy())

                current_section = {"title": line, "content": ""}
            else:
                current_section["content"] += line + "\n"

        # 添加最后一个章节
        if current_section["content"]:
            sections.append(current_section)

        return sections

    @staticmethod
    def estimate_complexity(outline: str) -> int:
        """估算大纲复杂度"""
        sections = OutlinePreprocessor.split_into_sections(outline)
        word_count = len(outline)

        complexity = len(sections) * 0.3 + min(word_count / 1000, 10) * 0.7
        return min(int(complexity * 10), 100)  # 0-100的复杂度评分