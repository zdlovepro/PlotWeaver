# 流水线模块边界

`pipeline/modules/` 中的数字就是唯一执行顺序。模块之间只能通过
`pipeline/contracts/` 中的公共契约和磁盘产物通信，禁止跨目录导入另一个
业务模块的内部 `service.py`、`parser.py` 或提示词。

| 模块 | 唯一职责 | 主要输入 | 主要输出 |
|---|---|---|---|
| 00 ingestion | 导入、切章、保留原文位置 | `input/` 小说文本 | `ChapterDocument` |
| 01 local synopsis | 理解局部情节并形成章节梗概 | `ChapterDocument` | `ChapterSynopsisBundle` |
| 02 hierarchical outline | 把章纲合并成故事弧、卷纲和全书纲 | `ChapterSynopsisBundle` | `HierarchicalOutlineBundle` |
| 03 fact hydration | 依据大纲反查并补充必要事实 | 分层大纲、章纲、原文 | `FactHydrationBundle` |
| 04 narrative graph | 合并实体、关系、状态、因果和跨章连续性 | 按需事实、大纲 | `NarrativeGraph` |
| 05 template mining | 归纳可迁移的情节和场景写法模板 | 大纲、图谱、多作品样本 | 模板库 |
| 06 style distillation | 归纳语言、节奏、视角和修辞策略 | 原文及结构对齐样本 | 作者风格档案 |
| 07 skill compiler | 将结构模板与风格策略编译成 Skill | 模板库、风格档案 | `{author_id}_skill/` |
| 08 generation | 在大纲、图谱和 Skill 约束下生成长篇正文 | 写作任务包 | 章节正文、图谱补丁 |
| 09 evaluation | 比较逻辑、内容和写法差距并给出改进信号 | 原文、生成文、所有中间态 | 质量报告 |

## 三条硬边界

1. 第一模块不得输出事实表、命题表、实体表或知识图谱。
2. 第二模块不得回到原文逐句提取信息，只能聚合第一模块的梗概。
3. 第三模块不得无差别扫描事实，只能围绕第二模块给出的事件、人物发展和
   未决线索提出事实需求，再回原文取证。

## 两种运行路线

- `short_validation`：5—20 章，第二模块最高合并到局部故事弧，然后转入第三模块。
- `full_book`：全书章纲分批合并到卷级和全书级，再从全书/卷纲反向分解事实需求。

两条路线共用同一个第一模块和第三模块，只在第二模块的最高聚合层不同。

