# 小说作者 Skill 产物规范

**状态：** 目标规范；标准 `draft` 目录、资格清单和可移植自检已实现，作者级核心与完整运行脚本待实现
**适用范围：** `output/<作者标识>_skill/` 及其编译、加载、验证过程

## 1. 目的

小说作者 Skill 不是一份长提示词，也不是把一部小说的设定和剧情换个名字保存下来。它是一个可移植、可追溯、可执行、可验证的写作能力包，回答四个问题：

1. 该作者在不同作品中稳定采用哪些叙事和语言选择；
2. 在给定故事状态、场景职责和篇幅时，作者通常怎样选择下一步；
3. 这些选择如何落成章节、场景和段落；
4. 如何判断生成结果既符合当前新故事事实，又比通用写法更接近目标作者。

作者标识由运行时传入。源码、注释和工程文档不得写死具体作者名称。用户输入 `ExampleAuthor` 时，输出目录为 `output/ExampleAuthor_skill/`；Skill 内部 `name` 使用由程序规范化后的英文小写连字符名称。

## 2. 能力边界

Skill 应包含：

- 跨作品稳定的作者核心特征；
- 单部作品相对作者核心的偏移；
- 相对同题材对照语料具有区分度的特征；
- 条件化叙事决策、长篇组织和正文实现策略；
- 检索、写作包组装、校验和候选重排脚本；
- 蒸馏证据摘要、数据版本、模型版本和盲测报告。

Skill 不应包含：

- 原始小说正文、长引文或可用于复写的连续文本；
- 来源作品的角色、地点、法宝、门派等专名库；
- 来源作品的完整剧情图谱或章节答案；
- 未经对照验证的题材共性；
- 仅凭单章或单部作品得出的“作者稳定规律”；
- API 密钥、绝对机器路径或只在编译机上可用的临时文件。

## 3. 产物层次

作者能力必须分层，禁止把所有观察混成一张风格总表。

| 层次 | 内容 | 能否进入作者核心 |
|---|---|---|
| `genre_baseline` | 同题材常见词汇、母题、升级结构、场景比例 | 否，仅作为扣除基线 |
| `author_core` | 跨多部作品稳定、相对对照作者有区分度的选择 | 是 |
| `work_delta` | 某部作品特有的节奏、视角、阶段偏移 | 否，作为可选适配层 |
| `context_profile` | 战斗、对话、发现、修炼、过渡等条件画像 | 是，但必须带适用条件 |
| `story_runtime` | 当前原创故事的事实、状态、因果和线索 | 否，由运行时外部提供 |

完整写作上下文按以下关系组装：

```text
作者核心 + 可选作品偏移 + 当前场景条件画像
+ 当前新故事的卷/事件/章节合同 + 当前图谱子图 + 已提交状态
```

## 4. 标准目录

```text
output/<作者标识>_skill/
├── SKILL.md
├── agents/
│   └── openai.yaml
├── references/
│   ├── author_core.json
│   ├── work_deltas.json
│   ├── genre_baseline.json
│   ├── narrative_policy.json
│   ├── prose_realization.json
│   ├── dialogue_patterns.json
│   ├── long_form_strategy.json
│   ├── negative_constraints.json
│   ├── quality_thresholds.json
│   └── evidence_report.json
├── scripts/
│   ├── select_style_context.py
│   ├── build_writer_packet.py
│   ├── validate_draft.py
│   ├── rank_candidates.py
│   └── self_check.py
└── evaluation/
    ├── evaluation_manifest.json
    └── acceptance_report.json
```

Skill 根目录不放安装说明、变更日志或项目README；这些内容保留在项目 `docs/`。`SKILL.md` 保持简洁，通过渐进加载引用 `references/`，不把所有 JSON 内容复制进主提示词。

当前 `draft` 包已经生成上述核心 references、`select_style_context.py`、`self_check.py`、提示词程序、`skill_manifest.json`、`skill_quality.json` 和 `evaluation/content_semantic_audit.json`。`build_writer_packet.py`、`validate_draft.py`、`rank_candidates.py` 及完整验收报告仍属于后续运行时阶段；目录结构存在不代表这些能力已经通过发布验收。

## 5. `SKILL.md` 契约

Frontmatter 只包含：

```yaml
---
name: <规范化作者标识>-novel-author
description: 使用已验证的小说作者写作策略，规划、生成、校验并续写中文长篇小说；适用于章节规划、场景写作、连续性维护和作者风格候选重排。
---
```

正文至少说明：

1. 何时加载哪一份参考资料；
2. 原创写作与结构保真评测的边界；
3. 卷级、事件级、章节级、场景级和段落级的执行顺序；
4. 故事图谱检索与作者策略检索的区别；
5. 候选生成、硬门控、风格重排、图谱补丁提交规则；
6. 失败时应返回的归因，而不是要求无限重试或自动降低标准。

## 6. 核心参考文件

### 6.1 `author_core.json`

每条作者核心特征至少包含：

```json
{
  "trait_id": "author-core:information-release:001",
  "family": "信息释放",
  "rule": "先呈现异常信号，再延迟解释，并让人物反应承担部分说明功能",
  "applicable_contexts": ["秘密发现", "危险预兆"],
  "counter_contexts": ["规则说明", "快速战斗结算"],
  "supporting_work_ids": ["work-001", "work-002"],
  "support_chapter_count": 12,
  "within_author_stability": 0.81,
  "genre_contrast_score": 0.67,
  "confidence": 0.76
}
```

只有满足跨作品支持和对照区分度的规则才能进入该文件。数值表示观测与置信度，不是要求生成模型机械凑数。

### 6.2 `narrative_policy.json`

它描述“在什么条件下选择什么写法”，而不是孤立的句式偏好。最低覆盖：

- 场景进入与退出；
- 目标—阻碍—行动—反应—后果；
- 冲突升级和代价配置；
- 信息释放与认知差；
- 对话中的权力变化；
- 章节钩子和转场；
- 多事件线程交替；
- 人物成长、资源变化和关系变化。

### 6.3 `long_form_strategy.json`

最低覆盖：卷级目标、阶段边界、事件依赖、章节职责、线程切换、状态积累和线索生命周期。它是作者长篇组织偏好，不保存来源作品的真实剧情答案。

### 6.4 `negative_constraints.json`

保存目标作者稳定避免、且生成模型容易出现的写法，例如无因果转场、重复总结、说明书式心理、无效感叹、没有代价的突然升级。每条负面规则也必须有样本或对照证据，不能凭开发者喜好硬编码。

## 7. 可执行脚本职责

| 脚本 | 职责 |
|---|---|
| `select_style_context.py` | 按场景职责、人物状态和叙事阶段选择少量相关作者规则。 |
| `build_writer_packet.py` | 合并层级大纲、当前故事子图、已提交状态和作者策略，产生受限写作包。 |
| `validate_draft.py` | 检查事实、状态、因果、时间空间、线索、文风和复写风险。 |
| `rank_candidates.py` | 先应用硬门控，再按作者辨识度、场景贴合、流畅度和新颖性重排。 |
| `self_check.py` | 检查目录、模式版本、依赖、引用路径和最低验收报告。 |

脚本应能在新环境中运行，或明确声明所需的 PlotWeaver 运行时版本。不能依赖编译机上的 `corpus/` 原文目录。

## 8. 状态与发布等级

| 状态 | 含义 |
|---|---|
| `draft` | 文件已生成，但仅有单作品或尚未完成语义抽检。 |
| `candidate` | 已完成多作品蒸馏和结构测试，但盲测或原创长篇测试尚未全部通过。 |
| `validated` | 跨作品、留出章节、原创长篇、作者辨识度和防复写测试全部通过。 |

当前编译器生成的包应标记为 `draft`，不得仅因 `skill_quality.json` 没有结构错误就标记为 `validated`。

## 9. 完成定义

一个 Skill 只有同时满足下列条件才算蒸馏成功：

1. 作者核心来自至少两部训练作品，并在未参与蒸馏的作品或连续章节上验证；
2. 所有作者核心规则有证据、适用条件、反例和对照区分度；
3. 生成器不读取来源原文，也能在留出任务中实现核心事件和状态变化；
4. 使用作者 Skill 比通用题材提示词在盲测中稳定提升作者辨识度；
5. 连续多章生成没有硬性状态矛盾、未来泄漏和未授权设定；
6. 防复写检查通过；
7. Skill 在独立目录中可加载、可执行、可自检，并带有明确版本与已知限制。
