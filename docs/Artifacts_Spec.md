# PlotWeaver JSON 产物结构说明书

**版本**：v1.0  
**适用分支**：`forth` 及后续重构分支  
**用途**：统一记录 PlotWeaver 每一步产生的 JSON 文件、字段结构、字段含义、上下游依赖和兼容策略。  

> 本文档专门约定“各步骤产物的 JSON 结构”。它不是代码实现文档，也不是 prompt 文档。后续新增字段时，应优先更新本文档。

---

## 0. 总体设计原则

### 0.1 Step Artifact 与 Pipeline State 分离

每一步应该同时区分两类产物：

1. **Step Artifact**：本步骤的干净产物，只包含本步骤负责生成的内容。
2. **Pipeline State Snapshot**：全局状态快照，可以包含前面所有步骤的累积结果。

错误示例：

```json
{
  "world_name": "...",
  "world_background": "...",
  "event_templates": [],
  "executable_templates": []
}
```

如果这是 `step7_templates.json`，就不应该混入 `world_background`。

正确做法：

```text
step5_world_base.json              # 只保存世界观
step6_interaction_patterns.json    # 只保存互动/关系/套路模式
step7_templates.json               # 只保存模板
pipeline_state_after_step7.json    # 保存全局状态快照
```

### 0.2 原始产物不可变

Step1 的物理切块产物应视为不可变 source chunk。Step2 可以生成新的 semantic window，但不能直接改写 Step1 的原始 chunk。

```text
Step1 chunk = 物理文本窗口
Step2 semantic_window = 情节完整窗口
Step2 plot_atom = 最小剧情单位
```

### 0.3 所有来源都必须可追踪

所有下游结构都应尽量携带：

```json
{
  "source_refs": [],
  "source_spans": [],
  "novel_sources": [],
  "arc_names": []
}
```

这样可以追踪某个模板、事件、大纲节点来自哪些原始素材。

### 0.4 JSON 输出格式约定

所有中间 JSON 文件建议使用：

```python
json.dump(payload, f, ensure_ascii=False, indent=2)
```

所有数组字段缺省为空数组，所有对象字段缺省为空对象，字符串字段缺省为空字符串。

---

## 1. 通用类型结构

### 1.1 SourceSpan

用于记录文本来源范围。

```json
{
  "chunk_id": "卷01_event0",
  "chapter_start": 1,
  "chapter_end": 6,
  "paragraph_start": 0,
  "paragraph_end": 0,
  "span_role": "current_chunk"
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | string | 来源 Step1 chunk ID |
| `chapter_start` | int | 起始章节 |
| `chapter_end` | int | 结束章节 |
| `paragraph_start` | int | 可选，起始段落 |
| `paragraph_end` | int | 可选，结束段落 |
| `span_role` | string | `current_chunk` / `carried_tail` / `complete_part` |

### 1.2 StateDelta

用于记录事件造成的状态变化。

```json
{
  "reputation": "+",
  "enemy_attention": "+",
  "resource": "+/-",
  "relationship": "changed",
  "injury": "optional",
  "secret_exposure": "optional",
  "open_hook": "+"
}
```

### 1.3 CoverageReport

用于记录候选挖掘和模板覆盖情况。

```json
{
  "atom_count": 0,
  "induced_event_count": 0,
  "candidate_count": 0,
  "cluster_count": 0,
  "final_count": 0,
  "covered_ratio": 0.0,
  "notes": []
}
```

### 1.4 JSON 文件命名建议

```text
intermediate_data/
  step1_chunks.json
  step2_semantic_windows.json
  step2_extracted_plots.json
  step2_tail_carry_log.json
  step3_induced_events.json
  step4_kb_manifest.json
  step5_world_base.json
  step6_interaction_patterns.json
  step7_templates.json
  pipeline_state_after_step7.json
  step8_skeleton_nodes.json
  step9_skeleton.json
  step10_casted_skeleton.json
  step11_reassembled_plot.json
  step12_volume_outlines.json
  step13_validation_result.json
```

---

# Step1：物理切块产物

## 2. Step1 作用

Step1 只负责原始文本的物理切块：章节识别、卷划分、长章节拆分、chunk ID 分配。它不负责识别情节原子。

## 2.1 输出文件

```text
intermediate_data/step1_chunks.json
```

## 2.2 顶层结构

建议按小说分组：

```json
{
  "novel_a.txt": [
    {
      "event_id": "卷01_event0",
      "source_chunk_id": "novel_a_卷01_chunk0001",
      "chunk_type": "physical",
      "novel_source": "novel_a.txt",
      "arc_name": "卷01",
      "chapter_start": 1,
      "chapter_end": 8,
      "chapters": [1, 2, 3, 4, 5, 6, 7, 8],
      "subchunk_index": 1,
      "subchunk_total": 1,
      "char_count": 12000,
      "summary": "",
      "text": "原文内容..."
    }
  ]
}
```

## 2.3 字段说明

| 字段 | 类型 | 是否必需 | 说明 |
|---|---|---:|---|
| `event_id` | string | 是 | 旧兼容字段，物理 chunk ID |
| `source_chunk_id` | string | 否 | 新推荐字段，物理 chunk ID |
| `chunk_type` | string | 是 | 固定为 `physical` |
| `novel_source` | string | 是 | 来源小说文件名 |
| `arc_name` | string | 是 | 卷名或分组名 |
| `chapter_start` | int | 是 | 起始章节 |
| `chapter_end` | int | 是 | 结束章节 |
| `chapters` | list[int] | 是 | 覆盖章节列表 |
| `subchunk_index` | int | 否 | 长章节拆分时的子块序号 |
| `subchunk_total` | int | 否 | 长章节拆分总数 |
| `char_count` | int | 否 | 文本字符数 |
| `summary` | string | 否 | 旧字段，Step1 可为空 |
| `text` | string | 是 | 原文内容 |

## 2.4 下游依赖

Step2 读取 Step1 chunk，但不能直接改写 Step1 文件。

---

# Step2：语义窗口、情节原子与 Tail 记录

## 3. Step2 作用

Step2 负责：

1. 将 Step1 物理 chunk 重组为 semantic window；
2. 从 semantic window 中抽取多个完整 PlotAtom；
3. 识别 incomplete tail；
4. 将 tail carry 到下一个 semantic window。

## 3.1 输出文件

```text
intermediate_data/step2_semantic_windows.json
intermediate_data/step2_extracted_plots.json
intermediate_data/step2_tail_carry_log.json
```

---

## 3.2 step2_semantic_windows.json

### 顶层结构

```json
[
  {
    "semantic_window_id": "sw_000001",
    "novel_source": "novel_a.txt",
    "arc_name": "卷01",
    "source_chunk_ids": ["卷01_event0"],
    "source_spans": [
      {
        "chunk_id": "卷01_event0",
        "chapter_start": 1,
        "chapter_end": 6,
        "span_role": "complete_part"
      }
    ],
    "text": "第1章到第6章文本...",
    "window_type": "normal",
    "contains_tail_from_previous": false,
    "carried_tail_id": "",
    "has_incomplete_tail": true,
    "incomplete_tail_id": "tail_000001",
    "notes": "第7-8章是下个事件开头，顺延到下一个窗口。"
  }
]
```

### 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `semantic_window_id` | string | Step2 语义窗口 ID |
| `novel_source` | string | 来源小说 |
| `arc_name` | string | 来源卷 |
| `source_chunk_ids` | list[string] | 由哪些 Step1 chunk 构成 |
| `source_spans` | list[SourceSpan] | 文本来源范围 |
| `text` | string | 实际送给 AI 的文本 |
| `window_type` | string | `normal` / `reflowed` |
| `contains_tail_from_previous` | bool | 是否包含上一个窗口 tail |
| `carried_tail_id` | string | 被合并进来的 tail ID |
| `has_incomplete_tail` | bool | 当前窗口尾部是否仍有未闭合 tail |
| `incomplete_tail_id` | string | 新 tail ID |
| `notes` | string | 人类可读说明 |

---

## 3.3 step2_extracted_plots.json

### 顶层结构

建议仍按小说分组，以兼容旧流程：

```json
{
  "novel_a.txt": [
    {
      "atom_id": "atom_000001",
      "semantic_window_id": "sw_000001",
      "atom_index_in_window": 1,
      "atom_type": "reversal",
      "novel_source": "novel_a.txt",
      "arc_name": "卷01",
      "chapter_start": 3,
      "chapter_end": 6,
      "chapter_count": 4,
      "source_spans": [
        {
          "chunk_id": "卷01_event0",
          "chapter_start": 3,
          "chapter_end": 6,
          "span_role": "complete_part"
        }
      ],
      "summary": "主角发现隐藏修炼方式并第一次反制挑衅。",
      "raw_summary": "AI 原始摘要...",
      "characters": ["叶尘", "血煞子"],
      "raw_characters": ["叶尘(主角)", "血煞子(压迫者)"],
      "character_keys": ["protagonist", "oppressor"],
      "location": "外门演武场",
      "raw_location": "外门演武场",
      "core_action": "主角借隐藏能力完成反制",
      "raw_core_action": "...",
      "cultivation_elements": ["凝气六层", "血炼禁制"],
      "actor": "叶尘",
      "goal": "证明自己并摆脱压迫",
      "obstacle": "血煞子设局并借规则压制",
      "action": "叶尘反用禁制证明对方设局",
      "outcome": "血煞子陷害失败，叶尘获得旁观者认可",
      "state_delta": {
        "reputation": "+",
        "enemy_attention": "+",
        "secret_exposure": "optional"
      },
      "causality_precondition": "前文血煞子持续压迫主角。",
      "causality_consequence": "血煞子开始更强烈地针对主角。",
      "motivation": "主角需要洗清嫌疑并争取资格。",
      "conflict_type": "规则压迫",
      "narrative_function": "公开反转",
      "emotion": "压抑后反转",
      "tension_level": 7,
      "is_complete": true,
      "boundary_reason": "该段有明确目标、阻碍、行动、结果和状态变化。"
    }
  ]
}
```

### PlotAtom 完整性要求

`is_complete = true` 的 PlotAtom 必须至少具备：

```text
summary
actor
goal
obstacle
action
outcome
state_delta
```

如果没有 `outcome` 或 `state_delta`，不应作为完整 PlotAtom 进入 Step3。

---

## 3.4 step2_tail_carry_log.json

### 顶层结构

```json
[
  {
    "tail_id": "tail_000001",
    "novel_source": "novel_a.txt",
    "arc_name": "卷01",
    "from_semantic_window_id": "sw_000001",
    "from_source_chunk_ids": ["卷01_event0"],
    "source_spans": [
      {
        "chunk_id": "卷01_event0",
        "chapter_start": 7,
        "chapter_end": 8,
        "span_role": "carried_tail"
      }
    ],
    "text": "第7-8章原文...",
    "summary": "主角刚接到秘境任务并抵达入口。",
    "reason": "新事件刚触发，尚未出现结果和状态变化。",
    "missing_parts": ["outcome", "state_delta"],
    "carry_to_next_chunk": true,
    "carried_to_semantic_window_id": "sw_000002",
    "status": "merged"
  }
]
```

### `status` 枚举

```text
open
merged
dropped_at_end
```

---

# Step3：事件归纳产物

## 4. Step3 作用

Step3 将多个 PlotAtom 归纳为更大的 InducedEvent。PlotAtom 是最小剧情单位，InducedEvent 是 3-10 章左右的事件块。

## 4.1 输出文件

```text
intermediate_data/step3_induced_events.json
```

## 4.2 顶层结构

```json
{
  "novel_a.txt": [
    {
      "event_id": "卷01_induced_event0",
      "novel_source": "novel_a.txt",
      "arc_name": "卷01",
      "source_atom_ids": ["atom_000001", "atom_000002"],
      "chapter_start": 1,
      "chapter_end": 6,
      "chapter_count": 6,
      "summary": "主角从低位受压到第一次公开反制，获得初步声望但引来更强敌意。",
      "raw_summary": "AI 原始事件归纳...",
      "conflict_hint": "规则压迫",
      "function_hint": "低位逆袭",
      "state_inputs": ["主角处于低位", "对手借规则压制"],
      "state_outputs": ["主角声望上升", "敌意增强"],
      "relationship_delta": ["血煞子对主角敌意升级"],
      "resource_delta": ["暴露血炼禁制"],
      "hook_open": ["血煞子背后势力尚未揭露"],
      "hook_close": [],
      "power_state": "凝气六层附近",
      "identity_state": "外门低位弟子",
      "character_keys": ["protagonist", "oppressor"]
    }
  ]
}
```

## 4.3 下游依赖

Step7 使用 InducedEvent 提取事件模板。Step8/Step9 使用 InducedEvent 或 SkeletonNode 构建骨架。

---

# Step4：知识库索引产物

## 5. Step4 作用

Step4 将 PlotAtom、人物信息、修炼元素、事件摘要等写入 RAG / ChromaDB 知识库。

## 5.1 建议输出文件

```text
intermediate_data/step4_kb_manifest.json
```

## 5.2 结构

```json
{
  "kb_backend": "chromadb",
  "persist_dir": "./chroma_db",
  "collections": {
    "events": {
      "count": 320,
      "source": "step2_extracted_plots.json"
    },
    "character_traits": {
      "count": 86,
      "source": "step2_extracted_plots.json"
    },
    "breakthrough_opportunities": {
      "count": 42,
      "source": "step2_extracted_plots.json"
    },
    "cultivation_systems": {
      "count": 12,
      "source": "step5_world_base.json"
    },
    "micro_interactions": {
      "count": 38,
      "source": "step6_interaction_patterns.json"
    }
  },
  "created_at": "",
  "notes": []
}
```

---

# Step5：世界观基础产物

## 6. Step5 作用

Step5 负责生成新世界观基础，不应混入 Step6/Step7 的互动模式或模板。

## 6.1 输出文件

```text
intermediate_data/step5_world_base.json
```

兼容旧文件：

```text
intermediate_data/step5_world_fusion.json
```

## 6.2 step5_world_base.json 结构

```json
{
  "world_name": "融合修真世界",
  "global_theme": "在失序修行体系中寻找新的道心秩序",
  "world_background": "...",
  "power_source": "灵气、血煞、神骸残响等融合后的新力量来源",
  "major_factions": ["天衡宗", "血煞盟", "边境商会"],
  "raw_system_text": "AI 原始世界观生成文本...",
  "cultivation_realms": [
    {
      "name": "凝气",
      "level": 1,
      "breakthrough_condition": "完成基础灵脉循环",
      "special_abilities": ["感知灵气", "基础术法"]
    }
  ]
}
```

## 6.3 禁止字段

`step5_world_base.json` 不应包含：

```text
macro_tropes
plot_threads
micro_interactions
event_templates
event_flow_templates
executable_templates
```

---

# Step6：互动模式、宏观套路与关系线

## 7. Step6 作用

Step6 负责从 PlotAtom 中挖掘人物互动模式、宏观剧情套路和长期关系线。它应采用候选挖掘 + 聚类合并，而不是固定抽 3-5 个。

## 7.1 输出文件

```text
intermediate_data/step6_interaction_patterns.json
```

兼容旧文件：

```text
intermediate_data/step6_interaction_mining.json
```

## 7.2 顶层结构

```json
{
  "micro_interaction_candidates": [],
  "macro_trope_candidates": [],
  "plot_thread_candidates": [],
  "micro_interaction_clusters": [],
  "macro_trope_clusters": [],
  "plot_thread_clusters": [],
  "micro_interactions": [],
  "macro_tropes": [],
  "plot_threads": [],
  "coverage_report": {
    "atom_count": 0,
    "micro_candidate_count": 0,
    "macro_candidate_count": 0,
    "thread_candidate_count": 0,
    "micro_cluster_count": 0,
    "macro_cluster_count": 0,
    "thread_cluster_count": 0,
    "final_micro_count": 0,
    "final_macro_count": 0,
    "final_thread_count": 0,
    "covered_atom_ratio": 0.0,
    "notes": []
  }
}
```

---

## 7.3 micro_interaction_candidate

```json
{
  "candidate_id": "micro_candidate_0001",
  "source_refs": ["atom_000001"],
  "novel_sources": ["novel_a.txt"],
  "arc_names": ["卷01"],
  "interaction_name": "压制试探后的局面翻转",
  "interaction_type": "试探-反制",
  "role_A": "压制者",
  "role_B": "被压制者/主角",
  "phase_sequence": ["试探", "施压", "误判", "反制", "余波"],
  "conflict_engine": "规则压迫",
  "relationship_delta": "敌意升级",
  "bystander_effect": "围观者态度从轻视转为震惊",
  "emotional_turn": "压抑到反转",
  "summary": "一方借规则试探压制，另一方克制观察后反制。"
}
```

## 7.4 macro_trope_candidate

```json
{
  "candidate_id": "macro_candidate_0001",
  "source_refs": ["atom_000001", "atom_000002"],
  "novel_sources": ["novel_a.txt"],
  "arc_name": "卷01",
  "name": "低位积累后集中翻盘",
  "abstract_arc": "从低位受压到公开证明自身价值",
  "stage_sequence": ["低位受压", "资源积累", "局部反击", "公开爆发", "余波升级"],
  "dominant_conflicts": ["规则压迫", "资源竞争"],
  "dominant_functions": ["低位逆袭", "主线推进"],
  "state_progression": {
    "reputation": "- -> +",
    "enemy_attention": "low -> high"
  },
  "summary": "主角先承受压力，再通过资源积累和局部反击完成公开翻盘。"
}
```

## 7.5 plot_thread_candidate

```json
{
  "candidate_id": "thread_candidate_0001",
  "source_refs": ["atom_000001", "atom_000009"],
  "novel_sources": ["novel_a.txt"],
  "character_keys": ["protagonist", "ally_01"],
  "display_characters": ["叶尘", "凌云"],
  "thread_type": "关系长线",
  "name": "从互相利用走向共同护持",
  "stage_sequence": ["试探", "互利合作", "共同负债", "信任重估"],
  "relationship_delta_sequence": ["陌生", "合作", "负债", "信任"],
  "support_count": 4,
  "summary": "两名角色从临时合作逐渐形成互相信任。"
}
```

## 7.6 cluster 通用结构

```json
{
  "cluster_id": "micro_cluster_0001",
  "cluster_key": "试探-反制|规则压迫",
  "candidate_ids": ["micro_candidate_0001", "micro_candidate_0007"],
  "source_refs": ["atom_000001", "atom_000007"],
  "support_count": 2,
  "novel_sources": ["novel_a.txt", "novel_b.txt"],
  "representative_candidate": {},
  "summary": "多个事件都呈现出压制试探后反制的互动结构。"
}
```

---

# Step7：模板挖掘产物

## 8. Step7 作用

Step7 负责将 PlotAtom、InducedEvent 和 Step6 模式进一步提炼为剧情模板库。最终目标不是保存原事件摘要，而是保存可填槽、可变形、可校验的 ExecutableTemplate。

## 8.1 输出文件

```text
intermediate_data/step7_templates.json
```

兼容旧文件：

```text
intermediate_data/step7_template_mining.json
intermediate_data/pipeline_state_after_step7.json
```

## 8.2 顶层结构

```json
{
  "template_candidates": [],
  "template_clusters": [],
  "role_slot_templates": [],
  "event_templates": [],
  "volume_templates": [],
  "event_flow_templates": [],
  "executable_templates": [],
  "coverage_report": {
    "atom_count": 0,
    "induced_event_count": 0,
    "template_candidate_count": 0,
    "template_cluster_count": 0,
    "final_event_template_count": 0,
    "final_flow_template_count": 0,
    "final_executable_template_count": 0,
    "covered_event_ratio": 0.0,
    "abstract_function_distribution": {},
    "conflict_engine_distribution": {},
    "notes": []
  }
}
```

## 8.3 template_candidate

```json
{
  "candidate_id": "template_candidate_0001",
  "source_refs": ["卷01_induced_event0"],
  "novel_sources": ["novel_a.txt"],
  "arc_names": ["卷01"],
  "source_type": "induced_event",
  "abstract_function": "公开反转",
  "conflict_engine": "规则压迫",
  "role_slot_signature": ["protagonist", "oppressor", "authority", "witness"],
  "beat_signature": ["公开压迫", "试探与误判", "反向破局", "代价与钩子"],
  "state_delta_signature": ["reputation:+", "enemy_attention:+", "secret_exposure:optional"],
  "required_preconditions": ["主角处于低位或被质疑", "存在公开规则场景"],
  "forbidden_source_details": ["宗门小比", "火灶房"],
  "source_pattern_summary": "低位主角在公开规则场景中被质疑，利用信息差完成反转。",
  "support_hint": 1
}
```

## 8.4 template_cluster

```json
{
  "cluster_id": "template_cluster_0001",
  "cluster_key": "公开反转|规则压迫|公开压迫>反向破局",
  "candidate_ids": ["template_candidate_0001", "template_candidate_0017"],
  "source_refs": ["卷01_induced_event0", "卷02_induced_event3"],
  "support_count": 2,
  "novel_sources": ["novel_a.txt", "novel_b.txt"],
  "abstract_function": "公开反转",
  "conflict_engine": "规则压迫",
  "merged_beat_signature": ["公开压迫", "试探与误判", "反向破局", "代价与钩子"],
  "merged_state_delta_signature": ["reputation:+", "enemy_attention:+", "open_hook:+"],
  "merged_role_slot_signature": ["protagonist", "oppressor", "authority", "witness"],
  "forbidden_source_details": ["宗门小比", "擂台打脸"],
  "representative_candidate": {},
  "is_rare_pattern": false
}
```

## 8.5 role_slot_template

```json
{
  "name": "trial_competition",
  "keywords": ["试炼", "考核", "大比", "斗法", "擂台", "选拔"],
  "role_slots": ["守关者", "竞争对手", "暗中使绊者", "旁观长辈"],
  "conflict_engines": ["名额争夺", "规则压制", "公开较量"]
}
```

## 8.6 event_template

```json
{
  "template_id": "event_template_0001",
  "name": "公开反转-规则压迫",
  "conflict_type": "规则压迫",
  "narrative_function": "公开反转",
  "role_slots": ["protagonist", "oppressor", "authority", "witness"],
  "conflict_engines": ["规则压迫"],
  "chapter_count_range": [3, 6],
  "sample_summaries": ["调试用短摘要，最多1-2条"]
}
```

## 8.7 event_flow_template

```json
{
  "template_id": "event_flow_0001",
  "name": "公开反转_4_chapters",
  "chapter_count_range": [4, 4],
  "dominant_conflicts": ["规则压迫"],
  "required_coverage": ["铺垫推进", "冲突升级", "局势转折", "代价落地", "下一事件钩子"],
  "beat_blueprint": [
    {
      "beat_index": 1,
      "chapter_span": [1, 1],
      "chapter_share": 0.25,
      "primary_function": "公开压迫",
      "secondary_functions": ["人物亮相"],
      "purpose": "让主角处于不利位置。"
    }
  ],
  "source_window_count": 2,
  "beat_count": 4
}
```

## 8.8 executable_template

```json
{
  "template_id": "executable_event_template_0001",
  "template_name": "公开规则压迫下的低位反转",
  "level": "event",
  "abstract_function": "公开反转",
  "conflict_engine": "规则压迫",
  "required_preconditions": [
    "主角处于低位或被质疑",
    "存在公开规则场景",
    "对手拥有规则、资源或舆论优势",
    "主角拥有隐藏能力、信息差或冒险策略"
  ],
  "role_slots": {
    "protagonist": "被压制者/破局者",
    "oppressor": "规则利用者",
    "authority": "裁决者",
    "witness": "舆论放大者",
    "ally": "有限支援者"
  },
  "beat_sequence": [
    {
      "beat_index": 1,
      "function": "公开压迫",
      "purpose": "让主角处于不利位置",
      "state_delta": {
        "reputation": "-",
        "pressure": "+"
      }
    },
    {
      "beat_index": 2,
      "function": "试探与误判",
      "purpose": "对手误判主角无力反击",
      "state_delta": {
        "enemy_confidence": "+"
      }
    },
    {
      "beat_index": 3,
      "function": "反向破局",
      "purpose": "主角利用隐藏能力或规则漏洞反转",
      "state_delta": {
        "reputation": "+",
        "secret_exposure": "optional"
      }
    },
    {
      "beat_index": 4,
      "function": "代价与钩子",
      "purpose": "胜利后引出更大风险",
      "state_delta": {
        "enemy_attention": "+",
        "open_hook": "+"
      }
    }
  ],
  "state_delta": {
    "reputation": "+",
    "enemy_attention": "+",
    "secret_exposure": "optional",
    "resource": "optional",
    "open_hook": "+"
  },
  "variation_axes": {
    "scene": ["审判庭", "任务复盘会", "商会验货", "遗迹资格审查", "城防听证"],
    "pressure_method": ["规则指控", "证据栽赃", "舆论围攻", "身份质疑"],
    "reversal_method": ["证据反推", "规则反用", "心理博弈", "第三方矛盾"],
    "cost": ["暴露底牌", "受伤", "背负债务", "失去信任", "被高层关注"]
  },
  "forbidden_source_details": ["宗门小比", "擂台打脸", "原功法名"],
  "source_refs": ["卷01_induced_event0", "卷02_induced_event3"],
  "source_pattern_summary": "低位主角在公开规则场景中被质疑或压制，利用隐藏能力、信息差或规则漏洞完成反转，获得认可但暴露底牌。"
}
```

## 8.9 禁止字段

`step7_templates.json` 不应包含：

```text
world_name
world_background
global_theme
power_source
cultivation_realms
macro_tropes
plot_threads
micro_interactions
```

---

# Step8：来源骨架抽取产物

## 9. Step8 作用

Step8 从来源小说事件中抽取 SkeletonNode，为 Step9 融合骨架做准备。

## 9.1 输出文件

```text
intermediate_data/step8_skeleton_nodes.json
```

## 9.2 顶层结构

```json
{
  "novel_a.txt": [
    {
      "node_id": "source_node_0001",
      "arc_name": "卷01",
      "realm_level": 1,
      "pacing_role": "开篇落点",
      "original_summary": "主角低位受压后第一次反制。",
      "conflict_hint": "规则压迫",
      "function_hint": "低位逆袭",
      "role_slots": ["protagonist", "oppressor", "authority"],
      "template_hint": "公开反转-规则压迫",
      "source_event_ids": ["卷01_induced_event0"],
      "chapter_start": 1,
      "chapter_end": 6,
      "chapter_count": 6,
      "chapter_blueprint": [],
      "character_keys": ["protagonist", "oppressor"],
      "source_novels": ["novel_a.txt"],
      "logic_card": {
        "preconditions": [],
        "state_outputs": [],
        "hook_open": [],
        "hook_close": [],
        "power_state": "凝气初期",
        "identity_state": "外门低位弟子"
      }
    }
  ]
}
```

---

# Step9：融合骨架产物

## 10. Step9 作用

Step9 从不同来源小说的 SkeletonNode 中融合出新小说的骨架。未来应以 StoryState 约束为核心，而不是只按节奏相似度拼接。

## 10.1 输出文件

```text
intermediate_data/step9_skeleton.json
```

## 10.2 结构

```json
{
  "base_novel": "novel_a.txt",
  "nodes": [
    {
      "node_id": "fused_event_0001",
      "arc_name": "卷01",
      "realm_level": 1,
      "pacing_role": "开篇落点",
      "original_summary": "低位主角在公开规则场景中完成第一次反转。",
      "adapted_summary": "",
      "conflict_hint": "规则压迫",
      "function_hint": "公开反转",
      "role_slots": ["protagonist", "oppressor", "authority"],
      "template_hint": "公开规则压迫下的低位反转",
      "source_event_ids": ["卷01_induced_event0"],
      "selection_ref": "novel_a.txt::卷01_induced_event0",
      "selection_source_novel": "novel_a.txt",
      "selection_source_index": 0,
      "chapter_start": 1,
      "chapter_end": 6,
      "chapter_count": 6,
      "chapter_blueprint": [],
      "character_keys": [],
      "source_novels": ["novel_a.txt"],
      "logic_card": {},
      "logic_notes": []
    }
  ],
  "character_sheet": null
}
```

## 10.3 logic_notes 建议枚举

```text
time_regression
growth_jump
identity_mismatch
location_gap
precondition_missing
hook_mismatch
power_mismatch
```

严重问题不应只记录，还应在后续进入 reject/repair/bridge 流程。

---

# Step10：角色 Cast 产物

## 11. Step10 作用

Step10 根据融合骨架和世界观生成新小说角色表。

## 11.1 输出文件

```text
intermediate_data/step10_casted_skeleton.json
```

## 11.2 结构

```json
{
  "base_novel": "novel_a.txt",
  "nodes": [],
  "character_sheet": {
    "protagonist": {
      "name": "叶尘",
      "role": "主角",
      "bond_depth": "核心",
      "entry_event": "fused_event_0001",
      "exit_event": "大结局",
      "return_event": "",
      "dao_heart": "不愿再被规则决定命运",
      "combat_style": "禁制与毒丹结合",
      "personality_flaw": "不信任权威",
      "background": "边境小城出身",
      "role_slots": ["protagonist"]
    },
    "supporting": [
      {
        "name": "凌云",
        "role": "有限支援者",
        "bond_depth": "卷级",
        "entry_event": "fused_event_0002",
        "exit_event": "fused_event_0009",
        "return_event": "",
        "dao_heart": "守住旧誓言",
        "combat_style": "剑术",
        "personality_flaw": "过度谨慎",
        "background": "宗门旁支",
        "role_slots": ["ally"]
      }
    ],
    "relationship_networks": [
      {
        "stage": "卷一前中段",
        "active_characters": ["叶尘", "凌云"],
        "relationship_status": "从试探合作到初步信任"
      }
    ],
    "event_role_plans": []
  }
}
```

---

# Step11：剧情重组产物

## 12. Step11 作用

Step11 将融合骨架节点重组为新世界观下的具体事件计划。

## 12.1 输出文件

```text
intermediate_data/step11_reassembled_plot.json
```

## 12.2 顶层结构

```json
[
  {
    "event_id": "adapted_fused_event_0001",
    "source_node_id": "fused_event_0001",
    "arc_name": "卷01",
    "realm_level": 1,
    "pacing_role": "开篇落点",
    "adapted_summary": "叶尘在任务复盘会上被执法弟子指控私藏禁物，他借污染痕迹反推证据来源，证明对方栽赃。",
    "event_plan": {
      "trigger": "执法弟子公开指控主角",
      "goal": "洗清嫌疑并保住任务资格",
      "obstacle": "执法堂掌握规则解释权",
      "reversal": "主角反用证据链证明指控者伪造报告",
      "cost": "暴露自己能感知污染残响",
      "target_chapter_count": 4,
      "chapter_blueprint": []
    },
    "active_characters": ["叶尘", "凌云", "执法弟子甲"],
    "state_updates": {
      "reputation": "+",
      "enemy_attention": "+",
      "secret_exposure": "+"
    },
    "logic_notes": [],
    "power_notes": [],
    "template_refs": ["executable_event_template_0001"]
  }
]
```

---

# Step12：分卷章节大纲产物

## 13. Step12 作用

Step12 将 Step11 的重组事件扩展为分卷章节级大纲。

## 13.1 输出文件

```text
intermediate_data/step12_volume_outlines.json
```

## 13.2 结构

```json
[
  {
    "volume_number": 1,
    "volume_title": "边城夜火",
    "realm_range": "凝气三层至凝气六层",
    "chapter_summaries": [
      "#### 剧情点：公开规则压迫下的低位反转",
      "**第1章**：叶尘在任务复盘会上被执法弟子公开质疑，旧任务报告成为压制他的证据。",
      "**第2章**：叶尘发现报告中的污染痕迹前后矛盾，却因身份低微无法直接反驳。",
      "**第3章**：他借现场残留的禁制气息反推出伪造者，使指控者陷入被动。",
      "**第4章**：叶尘保住资格，但暴露了异常感知能力，引来执法堂高层关注。"
    ],
    "tension_curve": "压迫 → 试探 → 反转 → 余波",
    "ending_state": "叶尘获得临时资格，但被执法堂列入观察名单。",
    "raw_text": "AI 原始返回文本"
  }
]
```

---

# Step13：验证和最终输出产物

## 14. Step13 作用

Step13 负责验证相似度、检查命名实体重合、输出最终 Markdown 大纲和世界圣经。

## 14.1 建议 JSON 输出文件

```text
intermediate_data/step13_validation_result.json
```

## 14.2 结构

```json
{
  "passed": false,
  "ner_overlap_ratio": 0.0741,
  "flagged_event_ids": ["adapted_fused_event_9"],
  "similar_tropes": ["公开规则场景中主角打脸对手并获高层认可"],
  "notes": "发现需要修改的节点，NER 重合率超过阈值。",
  "reports": {
    "world_bible_path": "output/novel_world_bible.md",
    "outline_path": "output/volume_1_to_3_outline.md",
    "validation_report_path": "output/validation_report.md"
  }
}
```

---

# PipelineStateSnapshot：全局状态快照

## 15. 用途

全局状态快照用于恢复和调试，不应代替某一步的干净产物。

## 15.1 输出文件示例

```text
intermediate_data/pipeline_state_after_step7.json
```

## 15.2 结构

```json
{
  "world_base": {
    "world_name": "",
    "global_theme": "",
    "world_background": "",
    "power_source": "",
    "major_factions": [],
    "cultivation_realms": [],
    "raw_system_text": ""
  },
  "interaction_patterns": {
    "micro_interactions": [],
    "macro_tropes": [],
    "plot_threads": [],
    "coverage_report": {}
  },
  "template_mining": {
    "template_candidates": [],
    "template_clusters": [],
    "executable_templates": [],
    "coverage_report": {}
  },
  "metadata": {
    "created_at": "",
    "pipeline_version": "",
    "notes": []
  }
}
```

---

# 各步骤产物依赖关系

```text
Step1 step1_chunks.json
  ↓
Step2 step2_semantic_windows.json
Step2 step2_extracted_plots.json
Step2 step2_tail_carry_log.json
  ↓
Step3 step3_induced_events.json
  ↓
Step4 knowledge base / step4_kb_manifest.json
  ↓
Step5 step5_world_base.json
  ↓
Step6 step6_interaction_patterns.json
  ↓
Step7 step7_templates.json
  ↓
Step8 step8_skeleton_nodes.json
  ↓
Step9 step9_skeleton.json
  ↓
Step10 step10_casted_skeleton.json
  ↓
Step11 step11_reassembled_plot.json
  ↓
Step12 step12_volume_outlines.json
  ↓
Step13 validation_report / final outline
```

---

# 检查脚本建议

## check_step2_atomization.py

检查：

```text
step2_semantic_windows.json 是否存在
step2_extracted_plots.json 是否存在
PlotAtom 是否有 semantic_window_id
is_complete=True 的 PlotAtom 是否有 outcome/state_delta
merged tail 是否有 carried_to_semantic_window_id
```

## check_step6_patterns.py

检查：

```text
step6_interaction_patterns.json 是否包含 candidates/clusters/final patterns
coverage_report 是否存在
候选数量不应固定为 3-5，除非素材极少
```

## check_step7_templates.py

检查：

```text
step7_templates.json 是否包含 template_candidates/template_clusters/executable_templates
executable_templates 字段是否完整
是否有乱码
是否混入 world_background/cultivation_realms 等非模板字段
```

## check_artifacts.py

检查干净产物边界：

```text
step5_world_base.json 不应包含模板字段
step6_interaction_patterns.json 不应包含世界观字段
step7_templates.json 不应包含世界观和 Step6 字段
```

---

# 版本演进建议

## v1.0

建立本文档，统一每一步 JSON 结构。

## v1.1

增加 JSON Schema 文件：

```text
schemas/step1_chunks.schema.json
schemas/step2_plot_atoms.schema.json
schemas/step7_templates.schema.json
```

## v1.2

引入 StoryState：

```json
{
  "protagonist": {
    "realm": "",
    "identity": "",
    "location": "",
    "items": [],
    "skills": [],
    "injuries": [],
    "known_info": []
  },
  "relationships": {},
  "open_hooks": [],
  "resolved_hooks": [],
  "active_conflicts": []
}
```

## v1.3

每个 Step 输出同时附带：

```json
{
  "schema_version": "1.3",
  "created_at": "",
  "producer_step": "Step7",
  "payload": {}
}
```

---

# 总结

本文档约定了 PlotWeaver 每一步 JSON 产物的边界和结构。核心原则是：

```text
Step1 只做物理切块；
Step2 生成 semantic window、PlotAtom 和 tail log；
Step5 只输出世界观；
Step6 输出候选、聚类和互动模式；
Step7 输出候选、聚类和可执行模板；
全局快照单独保存，不混入单步产物。
```

这样可以避免各步骤产物混杂，也能让后续状态机、模板驱动生成和区分度校验有稳定的数据基础。
