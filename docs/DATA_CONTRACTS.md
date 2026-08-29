# 模块数据契约

## 1. 规则

跨模块产物使用英文稳定字段、中文自然语言内容和显式 `schema_version`。模型可以
返回中文 JSON 键，但解析器必须在模块内部转换为公共契约。下游不得读取模型原始
回复、私有提示词字段或检查点。

## 2. 00 → 01：原文契约

`ChapterDocument` 保存不可变章节正文、来源哈希和自然段字符位置。第一模块可以
切出受限窗口，但不能改变原文字符和稳定段落 ID。

## 3. 01 → 02：梗概契约

```text
ChapterSynopsisBundle
  schema_version
  chapter_id
  source_hash
  local_synopses[]
    window_id
    reviewed_source_unit_ids[]
    opening_frame             # 仅首窗口非空；正文第一帧的中等粒度状态
    ending_frame              # 仅末窗口非空；正文最后一帧的中等粒度状态
    segments[]
      summary
      narrative_function
      story_change
      source_unit_ids[]
  chapter_synopsis
    one_sentence_summary
    chapter_function
    opening_state
    core_nodes[]
      summary
      narrative_function
      story_change
      local_segment_ids[]
    ending_state
    open_threads[]
    phases[]
  quality
```

当前契约版本为 `3.2`。此契约没有 `facts`、`claims`、`entities` 或知识图谱字段。段落 ID 只表示梗概依据；
`reviewed_source_unit_ids` 是程序生成的窗口覆盖元数据，不是逐段用途或事实抽取。
`opening_frame` 与 `ending_frame` 是章节压缩所需的两个边界句，不得扩展成人物状态账本；
章级解析器使用它们覆盖模型自由生成的首尾状态。

模块内部的章级状态终审使用补丁契约，而不是公共章纲契约的第二份副本：

```text
ChapterStateAuditPatch
  one_sentence_summary?       # null 表示不改
  core_node_replacements[]    # 只能替换既有 ID 的完整对象
  open_threads?               # null 表示不改；数组表示整体替换
  phase_replacements[]        # 只能替换既有 ID 的完整对象
```

该补丁无权修改章节作用、开篇状态、结尾状态，也无权新增、删除或重排节点。

## 4. 02 → 03：分层大纲契约

```text
HierarchicalOutlineBundle
  author_id
  work_id
  profile
  source_chapter_ids[]
  source_synopsis_hashes[]
  nodes[]
    outline_id
    level                    # story_arc / volume / book
    chapter_ids[]
    child_outline_ids[]
    summary
    opening_situation
    central_goal
    central_conflict
    causal_chain[]
    turning_points[]
    ending_change
    open_threads[]
    character_arcs[]
  root_outline_ids[]
  aggregation_ceiling
```

第二模块只能根据章纲聚合此结构，不携带整章原文或事实表。

## 5. 03 → 04：事实补全契约

```text
FactHydrationBundle
  outline_fingerprint
  requirements[]
    requirement_id
    outline_node_ids[]
    chapter_ids[]
    category
    question
    why_needed
    priority
  facts[]
    fact_id
    requirement_ids[]
    outline_node_ids[]
    chapter_id
    category
    statement
    subject_mentions[]
    related_mentions[]
    certainty
    evidence[]
    confidence
  unanswered_requirement_ids[]
```

事实必须同时回答一个明确需求、服务一个大纲节点并绑定原文证据。第三模块使用
原文称呼，不做跨章实体 ID 合并；全局实体、关系和状态归并属于第四模块。

## 6. 产物清单

每个阶段的 `manifest.json` 至少包含：

```json
{
  "module": "module_01_local_synopsis",
  "module_version": "3.23.0",
  "schema_version": "3.2",
  "profile": "short_validation",
  "input_hashes": {},
  "outputs": {},
  "model_policy": {},
  "accepted": true
}
```

## 7. 禁止耦合

- 第一模块不得导入第三模块或输出事实字段；
- 第二模块不得读取原文章节正文；
- 第三模块不得在没有 `FactRequirement` 的情况下增加事实；
- 第四模块不得从旧检查点绕过正式事实补全产物；
- 模板模块不得直接从原文章节生成作者模板；
- 下游不得从文件名猜测契约版本或读取 `accepted: false` 的产物。
