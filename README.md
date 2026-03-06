# PlotWeaver — 多小说情节混编系统

> 基于 RAG 技术，通过分析、拆解、重组多部小说的情节元素，自动生成全新小说大纲的智能写作辅助工具。

---

## 项目简介

PlotWeaver 是一个多小说情节混编（Plot Remix）系统，能够：

1. 读取多部小说文本（`.txt` 格式）
2. 自动提取情节单元，并进行叙事学标注
3. 分析情感曲线、张力曲线和因果关系图
4. 构建向量知识库（ChromaDB + RAG）
5. 通过人物映射和情节兼容性算法，将不同小说的情节元素融合重组
6. 调用大语言模型（DeepSeek 或 Ollama）生成完整的新小说大纲
7. 输出 Markdown 格式的新小说大纲文件

---

## 架构概览

```
输入小说 (.txt)
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1-2: 预处理                                               │
│  TextCleaner → ChapterSplitter → (并行批处理)                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: 情节提取  [LLM: extract_plots]                        │
│  PlotExtractor → raw_plots.json                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: 分析  [LLM: annotate_plot] + 纯算法                   │
│  NarrativeAnnotator → EmotionAnalyzer → TensionCurve            │
│                      → CausalGraph (NetworkX)                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: RAG 知识库                                             │
│  Embedder (sentence-transformers/OpenAI) → ChromaDB             │
│  MultiRetriever (4路检索 + 加权重排)                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 5: 人物映射  [纯算法]                                     │
│  CharacterMapper (余弦相似度 + 匈牙利算法)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 6: 情节混编  [纯算法]                                     │
│  CompatibilityScorer → ReplacementEngine (贪心)                  │
│                      → ConsistencyChecker (一致性验证)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 7: 大纲生成  [LLM: generate_outline]                     │
│  OutlineGenerator → OutlineFormatter → 新小说大纲.md            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
PlotWeaver/
├── src/
│   ├── __init__.py
│   ├── pipeline.py                    # 主流程编排（15个步骤）
│   ├── llm/
│   │   ├── __init__.py                # create_client() 工厂函数
│   │   ├── base.py                    # BaseLLMClient 抽象基类
│   │   ├── config.py                  # LLMConfig / DeepSeekConfig / OllamaConfig
│   │   ├── deepseek_client.py         # DeepSeek API 客户端
│   │   ├── ollama_client.py           # Ollama 本地模型客户端
│   │   ├── prompt_manager.py          # 3个提示词模板管理
│   │   └── response_parser.py         # LLM 响应 JSON 提取
│   ├── preprocess/
│   │   ├── text_cleaner.py            # UTF-8 编码处理、去噪
│   │   ├── chapter_splitter.py        # 按"第X章"/"Chapter X"切分
│   │   └── batch_processor.py         # 多小说并行预处理
│   ├── extraction/
│   │   └── plot_extractor.py          # LLM 情节单元提取
│   ├── analysis/
│   │   ├── narrative_annotator.py     # LLM 叙事标注
│   │   ├── emotion_analyzer.py        # 情感曲线分析（numpy）
│   │   ├── causal_graph.py            # 因果关系图（NetworkX）
│   │   └── tension_curve.py           # 张力曲线（scipy/numpy）
│   ├── rag/
│   │   ├── embedder.py                # 嵌入向量生成
│   │   ├── vector_store.py            # ChromaDB 持久化存储
│   │   └── multi_retriever.py         # 4路检索 + 加权重排
│   ├── remix/
│   │   ├── character_mapper.py        # 余弦相似度 + 匈牙利算法
│   │   ├── compatibility_scorer.py    # 加权兼容性评分
│   │   ├── replacement_engine.py      # 贪心替换算法
│   │   └── consistency_checker.py     # 因果完整性 / 人物一致性验证
│   ├── generation/
│   │   ├── outline_generator.py       # LLM 大纲生成
│   │   └── outline_formatter.py       # Markdown 格式化输出
│   └── models/
│       ├── plot_unit.py               # PlotUnit / AnnotatedPlotUnit
│       ├── character.py               # Character / CharacterMapping
│       ├── novel.py                   # Novel / Chapter
│       └── outline.py                 # Outline / ChapterOutline
├── config/
│   ├── pipeline.yaml                  # 全局配置
│   └── prompts/                       # 可选：自定义提示词覆盖
├── data/
│   ├── input/                         # 放置输入小说 .txt 文件
│   ├── intermediate/                  # 中间结果（JSON）
│   ├── vectordb/                      # ChromaDB 持久化数据
│   └── output/                        # 最终输出：新小说大纲.md
├── requirements.txt
└── README.md
```

---

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/zdlovepro/PlotWeaver.git
cd PlotWeaver
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 LLM 提供商

编辑 `config/pipeline.yaml`，选择 DeepSeek 或 Ollama：

**使用 DeepSeek API：**

```yaml
llm:
  provider: deepseek
  model: deepseek-chat      # 或 deepseek-reasoner
  api_key: "sk-your-api-key-here"
  temperature: 0.7
```

**使用 Ollama 本地模型：**

```bash
# 安装 Ollama: https://ollama.ai
ollama pull qwen2.5
```

```yaml
llm:
  provider: ollama
  model: qwen2.5            # 或 llama3, deepseek-r1 等
  ollama_host: "http://localhost:11434"
```

---

## 使用方法

### 快速开始

1. 将小说 `.txt` 文件放入 `data/input/` 目录（支持 UTF-8 和 GBK 编码）
2. 配置 `config/pipeline.yaml`
3. 运行主程序：

```bash
python main.py
```

或者在 Python 代码中使用：

```python
from src.pipeline import Pipeline

pipeline = Pipeline(config_path="config/pipeline.yaml")
output_path = pipeline.run()
print(f"大纲已生成: {output_path}")
```

### 指定输入文件

```python
from src.pipeline import Pipeline

pipeline = Pipeline()
output_path = pipeline.run(novel_files=[
    "data/input/小说A.txt",
    "data/input/小说B.txt",
    "data/input/小说C.txt",
])
```

### 单独使用各模块

```python
from src.llm import create_client, PromptManager

# 创建 LLM 客户端
client = create_client("ollama", model="qwen2.5")
response = client.ask("请用一句话介绍《红楼梦》的主题")
print(response)

# 检查 Ollama 可用性
print(client.is_available())
print(client.list_models())
```

```python
from src.preprocess import load_novel

novel = load_novel("data/input/my_novel.txt")
print(f"共 {len(novel.chapters)} 章")
for ch in novel.chapters[:3]:
    print(f"  {ch.title}: {len(ch.content)} 字")
```

---

## 15步流程说明

| 步骤 | 阶段 | 描述 | 是否调用LLM | 输出文件 |
|------|------|------|------------|----------|
| 1 | 预处理 | 读取并清洗小说文本 | 否 | — |
| 2 | 预处理 | 按章节切分 | 否 | `intermediate/{novel}/chapters.json` |
| 3 | 提取 | 情节单元提取 | **是** (`extract_plots`) | `intermediate/{novel}/raw_plots.json` |
| 4 | 分析 | 叙事学标注 | **是** (`annotate_plot`) | `intermediate/{novel}/annotated_plots.json` |
| 5 | 分析 | 情感曲线计算 | 否 | `intermediate/{novel}/emotion_curve.json` |
| 6 | 分析 | 张力曲线计算 | 否 | `intermediate/{novel}/tension_curve.json` |
| 7 | 分析 | 因果关系图构建 | 否 | `intermediate/{novel}/causal_graph.json` |
| 8 | RAG | 生成嵌入向量 | 否 | — |
| 9 | RAG | 构建 ChromaDB 知识库 | 否 | `vectordb/` |
| 10 | 人物映射 | 余弦相似度 + 匈牙利算法 | 否 | `intermediate/character_mapping.json` |
| 11 | 混编 | 计算兼容性矩阵 | 否 | `intermediate/compatibility_matrix.json` |
| 12 | 混编 | 贪心情节替换 | 否 | `intermediate/remixed_plots.json` |
| 13 | 混编 | 一致性验证 | 否 | `intermediate/validation_report.json` |
| 14 | 生成 | LLM 大纲生成 | **是** (`generate_outline`) | — |
| 15 | 生成 | Markdown 格式化输出 | 否 | `output/新小说大纲.md` |

> **LLM 仅在步骤 3、4、14 被调用**，其余步骤均为纯算法实现。

---

## 配置说明

`config/pipeline.yaml` 主要配置项：

```yaml
llm:
  provider: ollama          # "deepseek" 或 "ollama"
  model: qwen2.5
  temperature: 0.7
  api_key: "..."            # DeepSeek 使用
  ollama_host: "http://localhost:11434"

embedding:
  provider: local           # "local" 或 "openai"
  model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

rag:
  chromadb_path: "data/vectordb"
  top_k: 10

pipeline:
  framework_novel: ""       # 留空则使用第一部小说作为框架
  replacement_depth: 3      # 最多替换的情节单元数
  parallel_workers: 4       # 预处理并行线程数
```

---

## 设计原则

1. **LLM 仅用于 3 处任务**：情节提取、叙事标注、大纲生成
2. **其余全部为纯算法**：情感分析（numpy）、张力曲线（scipy）、因果图（NetworkX）、人物映射（余弦相似度 + 匈牙利算法）、兼容性评分（加权求和）
3. **支持两种 LLM 提供商**：DeepSeek（OpenAI 兼容 SDK）和 Ollama（httpx 直接 HTTP）
4. **中间结果全部持久化**：每步输出 JSON 文件，支持断点续跑
5. **多小说并行预处理**：Phase 1-3 支持 ThreadPoolExecutor 并行处理

---

## 许可证

MIT License
