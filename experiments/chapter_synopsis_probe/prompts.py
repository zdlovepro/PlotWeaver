from __future__ import annotations

import json
from typing import Any


EXTRACTION_PROMPT_REVISION = "1.0-pure-chapter-synopsis"
REVIEW_PROMPT_REVISION = "1.0-separated-chapter-reviews"

FIDELITY_DIMENSIONS = (
    "主体与对象",
    "确定程度",
    "信息来源",
    "因果关系",
    "流程状态",
    "时间与顺序",
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def extraction_prompt(*, chapter_id: str, title: str, source_text: str) -> str:
    return f"""任务：只根据本章正文，提取一段章节级梗概。

章节级梗概应概括本章从开始到结束的主要剧情推进，保留理解人物处境、关系、目标或后续方向发生变化所必需的主体、事件、因果和结果。区分已经发生的事实与人物言论、传闻、愿望、推测、计划和未完成过程。允许省略动作、对话、场景和气氛细节，但不得补充正文没有确认的信息。

这次请求只做梗概提取：不要复制证据，不要列出分析过程，不要评价自己的答案，不要拆成多个局部梗概。

严格输出且只输出以下JSON结构，不得增加字段：
{{
  "章节ID": {_json(chapter_id)},
  "章节梗概": "一段紧凑、连贯的章节级梗概"
}}

章节标题：{title}

本章完整正文：
<<<SOURCE_BEGIN>>>
{source_text}
<<<SOURCE_END>>>"""


def _review_input(
    *, chapter_id: str, title: str, source_rows: list[dict[str, str]], synopsis: str,
) -> str:
    return f"""章节ID：{chapter_id}
章节标题：{title}

待审章节梗概：
<<<SYNOPSIS_BEGIN>>>
{synopsis}
<<<SYNOPSIS_END>>>

带稳定段落ID的本章正文：
{_json(source_rows)}"""


def fidelity_review_prompt(
    *, chapter_id: str, title: str, source_rows: list[dict[str, str]], synopsis: str,
) -> str:
    checks = {dimension: "一致|不一致|无法判断" for dimension in FIDELITY_DIMENSIONS}
    return f"""任务：只审查章节梗概相对于正文的忠实性，不检查是否遗漏主要推进，也不检查篇幅或粒度。

分别核对主体与对象、确定程度、信息来源、因果关系、流程状态、时间与顺序。正文和梗概出现相同词语不能代替语义核对。只报告能够由正文定位的问题；不要修订梗概。

严格输出且只输出以下JSON结构，不得增加字段：
{{
  "目标ID": "CHAPTER_SYNOPSIS",
  "核对": {_json(checks)},
  "结论": "通过|需修订|无法判断",
  "问题": [
    {{
      "问题ID": "F1",
      "维度": "{FIDELITY_DIMENSIONS[0]}",
      "说明": "梗概与正文不一致的具体位置和原因",
      "相关原文段落ID": ["输入中真实存在的段落ID"]
    }}
  ]
}}

没有问题时“问题”必须是空数组。每个“不一致”维度至少对应一个问题；问题不能引用输入之外的段落ID。

{_review_input(chapter_id=chapter_id, title=title, source_rows=source_rows, synopsis=synopsis)}"""


def coverage_review_prompt(
    *, chapter_id: str, title: str, source_rows: list[dict[str, str]], synopsis: str,
) -> str:
    return f"""任务：只审查章节梗概是否覆盖本章主要剧情推进，不检查措辞级忠实性，也不检查细节多少。

主要推进是理解本章开始到结束的变化所必需的剧情：它改变人物处境、关系、目标、行动方向或后续剧情条件。不要因为普通动作、重复说明、气氛或场景细节没有写入而报遗漏。只判断重大遗漏、错误合并和顺序错误；不要修订梗概。

严格输出且只输出以下JSON结构，不得增加字段：
{{
  "目标ID": "CHAPTER_SYNOPSIS",
  "结论": "通过|需修订|无法判断",
  "问题": [
    {{
      "问题ID": "C1",
      "类别": "重大遗漏|错误合并|顺序错误",
      "说明": "缺失或组织错误的主要推进",
      "相关原文段落ID": ["输入中真实存在的段落ID"]
    }}
  ]
}}

没有问题时“问题”必须是空数组；问题不能引用输入之外的段落ID。

{_review_input(chapter_id=chapter_id, title=title, source_rows=source_rows, synopsis=synopsis)}"""


def granularity_review_prompt(
    *, chapter_id: str, title: str, source_rows: list[dict[str, str]], synopsis: str,
) -> str:
    return f"""任务：只审查章节梗概是否保持章节级的紧凑粒度，不检查事实忠实性，也不寻找遗漏的主要推进。

判断梗概是否复述了过多动作、对话、场景和过程细节，是否组织得过碎，或者是否把本应区分的推进压成了无法理解的一件事。只判断过度细节、过度拆分和错误合并；不要修订梗概。

严格输出且只输出以下JSON结构，不得增加字段：
{{
  "目标ID": "CHAPTER_SYNOPSIS",
  "结论": "通过|需修订|无法判断",
  "问题": [
    {{
      "问题ID": "G1",
      "类别": "过度细节|过度拆分|错误合并",
      "说明": "粒度问题的具体表现",
      "相关原文段落ID": ["输入中真实存在的段落ID"]
    }}
  ]
}}

没有问题时“问题”必须是空数组；问题不能引用输入之外的段落ID。

{_review_input(chapter_id=chapter_id, title=title, source_rows=source_rows, synopsis=synopsis)}"""
