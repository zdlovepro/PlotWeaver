# 新版流水线

## 执行顺序

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

核心方向是：先做章纲，再做跨章大纲，最后依据大纲反查必要事实。模块职责和原文访问权限见
[MODULE_BOUNDARIES.md](MODULE_BOUNDARIES.md)。

## 第一模块

第一模块按一个主导推进或一次主要状态变化划定连续事件单元，不按字符数或段落数切分。分区经过
独立粒度审校后，相邻完整块组成受限传输批次；每个分区恰好生成一条局部剧情梗概。每批只做一次按目标隔离的事实风险扫描，局部层不判断遗漏。警告允许通过；有效重大疑似错误需经独立裁决，成立后才最小修正一次，并只验证已确认问题。

锁定局部梗概后，首末块生成可追溯边界状态；章级阶段只读取全部有序局部梗概，形成允许多句的
完整章节梗概，不再增加节点分组层。章级事实一致性和主线完整性分别审查，`passed` 与 `has_warnings` 均由程序根据有效问题计算。正式契约为 `ChapterSynopsisBundle 5.0`。第一模块不输出事实、实体、关系、阶段、
剧情功能、跨章线程、时间线或图谱。

## 第二、三模块

第二模块按章节邻接和故事连续性聚合 `chapter synopsis → story arc → volume → book`，并从更高层
结构反向识别关键内容。第三模块只围绕大纲提出的明确需求回查原文并补全事实。

## 运行配置

短样本通常限制 20 章且最高聚合到 story arc；整本运行可聚合至 book。第一模块生产入口始终启用
语义审校，不提供跳过审校的 CLI 开关。独立模型能力实验放在 `experiments/`，不得写入正式产物。
