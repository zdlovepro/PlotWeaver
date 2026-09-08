# 第二模块：分层大纲

第二模块只接收第一模块的章节梗概，逐级合并为局部故事弧、卷级大纲和全书
大纲。它负责跨章事件链、人物发展线、长期目标、阶段冲突、关键转折和未决
线索，不负责回原文提取事实。

短样本运行在故事弧层停止；整本运行继续合并到卷级或全书级。最终产物是
第三模块反向提出事实需求的依据。

当前实现已经接通完整的语义层级：先逐对判断相邻章节的故事弧边界，再判断
相邻故事弧的语义卷边界；Book 不再判断边界，而是把全部有序 Volume 聚合为唯一
作品级根节点。每层节点分别进行忠实度与覆盖度审校，并做父子层整体复核。
单个节点最多修补一次；任何剩余阻断问题都会拒绝正式产物。

`StageContext.input_paths` 必须显式提供 `synopsis_manifest` 与 `synopsis_bundles`；
模块不会扫描并猜测最新运行，也不会在模型不可用时生成离线占位大纲。Book
只在模块一清单明确声明 `scope.complete_work=true` 时生成；部分输入可以停在
Story Arc 或 Volume，不能伪装成全书结论。

命令行入口：

```powershell
py main.py build-hierarchical-outline `
  --author-id AUTHOR_ID `
  --work-id WORK_ID `
  --source-run-id MODULE_01_RUN_ID `
  --aggregation-ceiling story_arc|volume|book
```
