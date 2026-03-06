from typing import Optional


# Built-in prompt templates (only 3)
_TEMPLATES: dict[str, dict] = {
    "extract_plots": {
        "system": (
            "你是一个专业的文学分析助手，擅长从小说文本中提取情节单元。"
            "请严格按照JSON格式输出，不要包含额外的解释。"
        ),
        "user": (
            "请从以下小说章节文本中提取所有独立的情节单元。\n\n"
            "章节文本：\n{chapter_text}\n\n"
            "请以JSON数组格式输出，每个情节单元包含以下字段：\n"
            "- content: 情节内容描述（50-200字）\n"
            "- characters: 涉及的人物列表\n"
            "- location: 发生地点\n"
            "- time_point: 时间点描述\n"
            "- event_type: 事件类型（冲突/转折/铺垫/高潮/结局/其他）\n\n"
            "输出格式：\n"
            '```json\n[{{"content": "...", "characters": ["..."], "location": "...", "time_point": "...", "event_type": "..."}}]\n```'
        ),
    },
    "annotate_plot": {
        "system": (
            "你是一个专业的叙事学分析助手，擅长分析小说情节的叙事功能、情感基调和因果关系。"
            "请严格按照JSON格式输出，不要包含额外的解释。"
        ),
        "user": (
            "请对以下情节单元进行叙事学标注。\n\n"
            "情节内容：\n{plot_content}\n\n"
            "涉及人物：{characters}\n"
            "事件类型：{event_type}\n\n"
            "请以JSON格式输出以下标注信息：\n"
            "- narrative_function: 叙事功能（开端/发展/转折/高潮/结局/铺垫/悬念/回忆）\n"
            "- emotion_tone: 情感基调（积极/消极/紧张/平静/悲伤/喜悦/愤怒/恐惧）\n"
            "- tension_level: 张力值（0.0-1.0的浮点数）\n"
            "- conflict_type: 冲突类型（人物冲突/人与自然/内心冲突/社会冲突/无冲突）\n"
            "- causal_summary: 因果关系摘要（描述此情节的前因后果，50字以内）\n"
            "- themes: 主题标签列表（如：['爱情', '复仇', '成长']）\n\n"
            "输出格式：\n"
            '```json\n{{"narrative_function": "...", "emotion_tone": "...", "tension_level": 0.5, '
            '"conflict_type": "...", "causal_summary": "...", "themes": ["..."]}}\n```'
        ),
    },
    "generate_outline": {
        "system": (
            "你是一个专业的小说策划师，擅长创作新颖的小说大纲。"
            "你将基于多部小说的情节元素，创作一个融合重组的全新小说大纲。"
            "请严格按照JSON格式输出，确保故事逻辑连贯、人物性格鲜明。"
        ),
        "user": (
            "请根据以下重组后的情节素材，生成一部新小说的完整大纲。\n\n"
            "参考情节素材：\n{remixed_plots}\n\n"
            "人物映射关系：\n{character_mapping}\n\n"
            "情感曲线要求：\n{emotion_curve_requirement}\n\n"
            "张力曲线要求：\n{tension_curve_requirement}\n\n"
            "请以JSON格式输出完整的小说大纲，包含以下内容：\n"
            "- title: 新小说标题\n"
            "- premise: 故事前提（100字以内）\n"
            "- theme: 核心主题\n"
            "- chapters: 章节大纲列表，每章包含：\n"
            "  - number: 章节序号\n"
            "  - title: 章节标题\n"
            "  - summary: 章节内容摘要（100-200字）\n"
            "  - key_events: 关键事件列表\n"
            "  - characters: 出场人物列表\n"
            "  - emotion_tone: 情感基调\n"
            "  - tension_level: 张力值（0.0-1.0）\n\n"
            "输出格式：\n"
            '```json\n{{"title": "...", "premise": "...", "theme": "...", '
            '"chapters": [{{"number": 1, "title": "...", "summary": "...", '
            '"key_events": ["..."], "characters": ["..."], "emotion_tone": "...", "tension_level": 0.5}}]}}\n```'
        ),
    },
}


class PromptManager:
    """Manages the 3 built-in LLM prompt templates with optional file-based overrides."""

    def __init__(self, prompts_dir: Optional[str] = None):
        self._templates = dict(_TEMPLATES)
        if prompts_dir:
            self._load_overrides(prompts_dir)

    def _load_overrides(self, prompts_dir: str) -> None:
        import os
        import yaml  # type: ignore[import]
        for name in self._templates:
            filepath = os.path.join(prompts_dir, f"{name}.yaml")
            if os.path.isfile(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        override = yaml.safe_load(f)
                    if isinstance(override, dict):
                        self._templates[name].update(override)
                except Exception:
                    pass

    def get_system(self, template_name: str) -> str:
        if template_name not in self._templates:
            raise KeyError(f"Unknown prompt template: {template_name!r}. Available: {list(self._templates)}")
        return self._templates[template_name]["system"]

    def get_user(self, template_name: str, **kwargs) -> str:
        if template_name not in self._templates:
            raise KeyError(f"Unknown prompt template: {template_name!r}. Available: {list(self._templates)}")
        return self._templates[template_name]["user"].format(**kwargs)

    def list_templates(self) -> list[str]:
        return list(self._templates.keys())
