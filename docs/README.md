# PlotWeaver 文档索引

当前文档以事件优先、分层大纲、按需事实和模块解耦为基线。旧的事实密度先行流程不再代表目标架构。

## 推荐阅读顺序

1. [SRS.md](SRS.md)：系统要解决什么问题。
2. [ARCHITECTURE.md](ARCHITECTURE.md)：模块、目录和公开/私有边界。
3. [MODULE_BOUNDARIES.md](MODULE_BOUNDARIES.md)：00—09 各模块的输入、职责、禁止事项和验收边界。
4. [PIPELINE.md](PIPELINE.md)：两条路线与执行顺序。
5. [MODULE_01_SPEC.md](MODULE_01_SPEC.md)：第一模块的唯一正式定义。
6. [DATA_CONTRACTS.md](DATA_CONTRACTS.md)：跨模块 JSON 契约。
7. [SDS.md](SDS.md)：软件设计细节。
8. [EVALUATION_SPEC.md](EVALUATION_SPEC.md)：质量和发布门槛。
9. [AUTHOR_SKILL_SPEC.md](AUTHOR_SKILL_SPEC.md)：最终 Skill 产物。
10. [MODEL_GUIDE.md](MODEL_GUIDE.md)：API、本地模型、RAG 和可选微调。
11. [PROMPT_CONVENTIONS.md](PROMPT_CONVENTIONS.md)：项目级提示词结构、DeepSeek 适配和评测规范。
12. [PRIVATE_MODULE.md](PRIVATE_MODULE.md)：第一模块和语料的 Git 边界。

## 当前工程结论

- 第一模块是私有的中等粒度剧情压缩模块，不包含逐段动作账本或事实提取。
- 短样本与整本提取不是两套代码，只改变分层聚合的转折点。
- 多重模板仍然保留，但必须消费已验证的事件、大纲和图谱。
- 最终目标是跨作品蒸馏小说作者的可复用写作 Skill，不是保存某一本书的情节摘要。
