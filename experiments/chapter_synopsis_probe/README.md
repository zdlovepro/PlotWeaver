# 章节梗概提取能力独立实验

这个实验只回答一个问题：模型直接读取完整章节时，能否生成正确、完整且粒度合适的章节级梗概。

它不会运行第一模块，不读取第一模块的局部梗概或检查点，也不会修改正式流水线产物。脚本通过 `pipeline.common.model.complete_json` 使用项目统一的模型配置，不自行实现HTTP或SDK调用。

## 运行

```powershell
py -3 experiments/chapter_synopsis_probe/run_probe.py `
  --author-id ErGen `
  --work-id work-001-clean `
  --limit 3 `
  --run-id probe-001 `
  --audit full
```

只生成梗概、不调用独立审校：

```powershell
py -3 experiments/chapter_synopsis_probe/run_probe.py `
  --author-id ErGen `
  --work-id work-001-clean `
  --chapter 1,3,5 `
  --run-id extraction-only-001 `
  --audit none
```

每个请求固定 `attempts=1` 且禁用思考模式回退。非法JSON、模型服务错误和审校协议错误都只记录，不会补发请求或修订梗概。已经存在的 `run-id` 会被拒绝，避免新旧结果混合。

## 调用数量

- `--audit none`：每章一次提取调用。
- `--audit full`：每章一次提取调用，之后分别进行忠实性、主要推进覆盖和章节粒度审校，共四次调用。

三类审校互相独立。提取请求只返回章节梗概，不要求证据、分析或自我评价。

## 结果目录

结果写入 `experiments/chapter_synopsis_probe/runs/<run-id>/`。每章保存：

- `source.json`：本章完整原文和稳定段落ID。
- `*.prompt.txt`：带真实换行、便于人工阅读的模型输入。
- `*.request.json`：模型、模式和实际 system/user 消息。
- `*.raw.txt`：模型未经解析的原始返回。
- `*.attempt.json`：返回元数据或调用错误。
- `*.result.json`：成功解析出的JSON对象。
- `decision.json`：协议与内容结论。

运行目录根部保存 `manifest.json`、`summary.json` 和 `report.md`。任何产物都不会保存API密钥。

## 状态

- `passed`：三类独立审校全部通过。
- `unreviewed`：只完成提取，未做模型审校。
- `content_failed`：至少一类审校确认存在内容问题。
- `inconclusive`：审校无法判断或审校集合不完整。
- `extraction_protocol_failed`：提取返回不符合最小JSON协议。
- `review_protocol_failed`：至少一个审校返回不符合自己的协议。
- `model_service_failed`：模型服务请求失败。
- `source_too_large`：章节超过显式上限；脚本不会静默截断。
