# PlotWeaver 目标架构

## 1. 架构目的

PlotWeaver 从同一小说作者的多部作品中提取可复用的故事组织规律和正文写法，
再编译为可安装、可验证的小说作者 Skill。系统先恢复“故事讲了什么以及为什么
这样发展”，随后才回到原文补充支撑这些结构的必要事实。

## 2. 唯一数据流

```text
原始文本
  ↓ 00 导入、切章、稳定位置
不可变 ChapterDocument
  ↓ 01 受限窗口理解、章内合并
局部剧情段 → 章节核心节点 + 章节阶段
  ↓ 02 跨章逐级聚合
故事弧大纲 → 卷级大纲 → 全书大纲
  ↓ 03 大纲提出事实需求，再回原文取证
必要人物/关系/时空/状态/因果/线索事实
  ↓ 04 统一身份、状态和跨章连接
narrative_graph.json
  ↓ 05—07
多重模板 + 风格档案 → 小说作者 Skill
  ↓ 08—09
受控正文生成 → 结构、逻辑、文风评测
```

不存在“第一模块事实层”或“引用层”。第一模块只输出梗概；段落 ID 只是溯源
索引，不构成新的内容层。事实从第三模块开始出现。

## 3. 活动目录

```text
pipeline/
  cli.py
  common/                         # JSON、模型、路径等通用能力
  contracts/                      # 跨模块稳定数据契约
    source.py                     # 00：不可变原文
    synopsis.py                   # 01：局部剧情段/章节核心节点/章节阶段
    outline.py                    # 02：故事弧/卷/全书大纲
    fact_hydration.py             # 03：大纲驱动的事实需求与证据事实
    graph.py                      # 04：叙事图谱
    ...                           # 后续模块契约
  orchestrator/                   # 阶段加载与运行上下文
  modules/
    module_00_ingestion/
    module_01_local_synopsis_private/
    module_02_hierarchical_outline/
    module_03_fact_hydration/
    module_04_narrative_graph/
    module_05_template_mining/
    module_06_style_distillation/
    module_07_skill_compiler/
    module_08_generation/
    module_09_evaluation/
```

`v2/`只作为历史思路参考，不被新版入口导入。放错位置的事实优先实现已移到
`.trash/`，不会被活动流水线加载。

## 4. 模块边界

本节给出摘要；完整的输入、原文访问权限、禁止事项和验收条件以
[MODULE_BOUNDARIES.md](MODULE_BOUNDARIES.md) 为准。

| 模块 | 负责 | 明确不负责 |
|---|---|---|
| 00 | 文本导入、切章、原文位置 | 情节理解 |
| 01 | 中等粒度局部剧情段、章节核心节点和章节阶段 | 逐段动作账本、事实表、实体表、跨章大纲 |
| 02 | 故事弧、卷、全书大纲及长程因果 | 回原文逐句取证 |
| 03 | 根据大纲按需提取证据事实 | 无差别事实覆盖、实体全局合并 |
| 04 | 实体合并、关系、时空、状态、因果、线索生命周期 | 文风蒸馏 |
| 05 | 宏观、事件、场景、人物、节奏、悬念模板 | 语言风格 |
| 06 | 语言、视角、节奏、信息释放和修辞策略 | 原作剧情复制 |
| 07 | Skill 组织、提示词程序和发布资格 | 正文生成 |
| 08 | 受控长文生成与图谱补丁 | 用未来原文提示生成器 |
| 09 | 内容、逻辑、文风差距与错误归因 | 通过降低标准掩盖失败 |

每个模块只能导入 `pipeline.common`、`pipeline.contracts` 和自身代码。模块间通过
版本化磁盘产物通信，不允许导入其他业务模块的内部提示词、解析器或服务。

## 5. 两条运行路线

### 5.1 短样本验证

- 连续处理 5—20 章；
- 第一模块逐章生成同样的章纲；
- 第二模块合并到局部故事弧后停止；
- 从故事弧向下生成事实需求；
- 只能验证程序，不能宣称获得稳定作者 Skill。

### 5.2 整本提取

- 所有章节仍然按受限窗口处理；
- 第二模块分批合并到卷级和全书级；
- 从全书或卷级大纲向下生成事实需求；
- 再通过多作品对照区分作者规律与作品特例。

两条路线只在第二模块的聚合上限不同，不维护两套局部提取或事实提取代码。

## 6. 公共和私有边界

第一模块的提示词、解析和内容质量策略位于：

```text
pipeline/modules/module_01_local_synopsis_private/
```

该目录不上传公共 Git。公共仓库保留 `contracts/synopsis.py`、动态入口、接口文档
和不含小说原文的合成契约测试。公共文档和代码不得写死具体作者姓名。

## 7. 产物布局

```text
runs/{author_id}/{run_id}/
  module_00_ingestion/
  module_01_local_synopsis/
  module_02_hierarchical_outline/
  module_03_fact_hydration/
  ...
```

每阶段至少包含 `manifest.json`、正式输出、`quality_report.json`、错误记录和可恢复
检查点。下游只能读取清单中标记为接受的上游产物。
