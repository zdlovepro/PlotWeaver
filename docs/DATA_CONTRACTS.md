# 模块数据契约

## 1. 通用规则

跨模块产物使用英文稳定字段、中文自然语言内容和显式 `schema_version`。模型可返回中文 JSON
键，解析器必须在模块内部转换。下游不得读取原始模型回复、私有提示词或检查点。

## 2. 00 → 01

`ChapterDocument` 保存不可变正文、来源哈希和稳定自然段 ID。第一模块可建立私有导航范围，
但不能改写原文或段落 ID。

## 3. 01 → 02：schema 5.0

```text
ChapterSynopsisBundle
  schema_version
  chapter_id
  source_hash
  local_synopses[]
    window_id
    chapter_id
    reviewed_source_unit_ids[]
    segments[]
      segment_id
      order
      summary
      source_unit_ids[]
  chapter_synopsis
    chapter_id
    opening_state { text, source_unit_ids[] }
    chapter_summary { text, local_segment_ids[] }
    ending_state { text, source_unit_ids[] }
  quality
    passed
    content_status
    source_window_coverage
    local_synopsis_count
    local_segment_count
    issues[]
```

当前版本为 `5.0`。它不包含事实、实体、关系、剧情功能、变化字段、章节节点、章节阶段或跨章未决线程。
段落 ID 仅用于回查梗概依据。每个 `local_synopses[]` 对应一个经过审校的语义分区，且其
`segments[]` 必须恰好包含一条局部事件梗概；分区数量和尺寸由语义结构决定，不设固定上限。
`chapter_summary` 允许包含多个句子，并按原文顺序直接引用全部局部梗概ID。

## 4. 后续契约

第二模块只读取已接受的 schema 5.0 章纲并聚合事件与跨章结构。第三模块根据跨章大纲提出
`FactRequirement`，再回查原文生成 `FactHydrationBundle`。第一模块不得提前承担这两步。

## 5. Manifest

每个阶段的 `manifest.json` 至少包含 `module`、`module_version`、`schema_version`、
`profile`、`input_hashes`、`outputs`、`model_policy` 和 `accepted`。下游不得读取
`accepted: false` 的正式产物。
