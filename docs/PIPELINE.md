# 新版流水线

## 1. 执行顺序

```text
00 ingest
01 extract-synopsis
02 build-hierarchical-outline
03 hydrate-required-facts
04 build-narrative-graph
05 mine-templates
06 distil-style
07 compile-skill
08 generate
09 evaluate
```

前三个智能阶段的方向不可交换：

```text
先做章纲 → 再做跨章大纲 → 最后依据大纲反查事实
```

完整模块职责与原文访问权限见 [MODULE_BOUNDARIES.md](MODULE_BOUNDARIES.md)。

## 2. 第一模块

入口目录：

```text
pipeline/modules/module_01_local_synopsis_private/
```

产物契约：`ChapterSynopsisBundle`。每个局部模型读取受限原文窗口，并压缩为
0—3 个局部剧情段；首、末窗口各额外固定一个章首/章末边界句。章级模型只读取
这些局部结果和边界句，形成至多 8 个核心剧情节点和至多 6 个章节
阶段，并且无权改写边界句。第一模块不输出逐段动作账本、事实、命题、实体或图谱。

## 3. 第二模块

第二模块只读取章节梗概，按章节邻接和故事连续性分批聚合：

```text
chapter synopsis → story arc → volume → book
```

短样本最高到 `story_arc`，整本最高到 `book`。第二模块输出的目标、冲突、因果、
转折和人物发展线会成为第三模块的事实需求来源。

## 4. 第三模块

第三模块首先针对每个大纲节点提出具体问题，例如：

- 哪个已知目标促成了这次选择？
- 哪项限制使人物不能采用另一条路线？
- 行动后人物、关系、资源、位置或认知发生了什么变化？
- 这条跨章因果需要哪些原文证据？
- 一个开放线索在样本后文是否得到回应？

只有回答这些问题所需的信息才进入事实补全。每条事实必须关联事实需求、大纲
节点、章节和原文证据。

## 5. 后续阶段

第四模块把按需事实统一成叙事图谱；第五和第六模块分别提取结构模板和文风；
第七模块编译 Skill；第八模块通过段落写作包生成正文并提交图谱补丁；第九模块
比较内容、逻辑、写法和阅读质量，并把错误归因到具体模块。

## 6. 运行配置

短样本：

```yaml
profile: short_validation
chapter_limit: 20
aggregation_ceiling: story_arc
allow_final_skill: false
```

整本：

```yaml
profile: full_book
aggregation_ceiling: book
require_cross_work_templates: true
allow_final_skill: true
```

## 7. 迁移状态

- 00 模块可用；
- 01 已按中等粒度剧情压缩边界重建契约、提示词、服务和质量门槛；当前契约为 3.2，
  首尾边界由直接阅读边界原文的局部阶段锁定，正在进行真实语料验收；
- 02、03 已建立正确契约和入口边界，未接通前必须显式报错；
- 旧事实优先和旧图谱反推大纲代码已移出活动流水线并保留可恢复备份；
- 04—09 的旧实现仅在其输入契约迁移完成后重新接入。
