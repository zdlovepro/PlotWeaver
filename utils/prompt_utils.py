import json


class PromptCreator:
    @staticmethod
    def create_prompt(original_outline, insert_outline, insertion_points):
        """创建融合提示"""
        prompt = f"""请将以下两个小说大纲融合成一个新的、逻辑连贯的大纲。

原大纲（基础）:
{original_outline}

要插入的情节大纲:
{insert_outline}

插入点和修改要求:
{json.dumps(insertion_points, ensure_ascii=False, indent=2)}

请按照以下要求生成新大纲:
1. 以原大纲为主线框架
2. 在指定位置插入第二个大纲的情节元素
3. 保持故事逻辑的连贯性和合理性
4. 可以适当修改原大纲的主线来适应新情节
5. 输出格式为JSON，包含title、main_plot、sub_plots、characters、key_events等字段

生成的新大纲:
"""
        return prompt

    @staticmethod
    def extract_generated_outline(full_text):
        """提取生成的大纲内容"""
        if "生成的新大纲:" in full_text:
            return full_text.split("生成的新大纲:")[-1].strip()
        elif "生成的新大纲" in full_text:
            return full_text.split("生成的新大纲")[-1].strip()
        else:
            return full_text