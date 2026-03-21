# PlotWeaver V2.0

An automated Python pipeline for fusing multiple Xianxia (Chinese cultivation) novel outlines into a new, highly coherent, and plagiarism-resistant novel outline.

## Features

- **7-step automated pipeline** powered by DeepSeek API and ChromaDB
- Semantic arc detection (no fixed character-count chunking)
- Dual-stage plot extraction (objective facts + subjective logic)
- RAG knowledge base with 4 ChromaDB collections
- Fused cultivation system generation with NetworkX DAG validation
- Character-driven plot reassembly with personality alignment
- Sliding window volume generation (avoids token-limit truncation)
- Adversarial plagiarism check (NER exact-match + DeepSeek anti-plagiarism editor)

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure**
   - Edit `config.yaml` and set your `DEEPSEEK_API_KEY`, or set the environment variable:
     ```bash
     export DEEPSEEK_API_KEY=your_key_here
     ```
   - Set `input_dir` (default `./input`), `output_dir` (default `./output`), and
     `intermediate_dir` (default `./intermediate_data`).

3. **Add source novels**
   - Place your source novel `.txt` files in the `input` directory.

4. **Run**
   ```bash
   python main.py
   ```

## Resuming the Pipeline (`--start-step`)

If the pipeline is interrupted (e.g., network error, API timeout), you can resume
from any step without re-running earlier, expensive steps.

After Step 1 and Step 2 complete successfully, their outputs are automatically saved
as JSON files inside `intermediate_dir` (default `./intermediate_data`):

| File | Written by | Contents |
|------|-----------|---------|
| `intermediate_data/step1_chunks.json` | Step 1 | Volume arcs & semantic chunks for every source novel |
| `intermediate_data/step2_extracted_plots.json` | Step 2 | Extracted objective & subjective plot atoms |

### Resume examples

```bash
# Full run (default)
python main.py

# Resume from Step 2 – loads step1_chunks.json, skips Step 1
python main.py --start-step 2

# Resume from Step 3 – loads step2_extracted_plots.json, skips Steps 1 & 2
python main.py --start-step 3

# Show all options
python main.py --help
```

### `--help` output

```
usage: python main.py [--start-step N] [--help]

PlotWeaver V2.0 – Xianxia Novel Outline Fusion Pipeline

options:
  -h, --help       show this help message and exit
  --start-step N   Step number to start from (1-7). Steps before N are skipped
                   and their outputs are loaded from intermediate_data/.
                   Default: 1 (full run).
```

## Output

After a successful run, the `output` directory will contain:

| File | Description |
|------|-------------|
| `novel_world_bible.md` | New cultivation system, protagonist sheet, character relationships |
| `volume_1_to_N_outline.md` | Full per-chapter outline split by cultivation arc |
| `validation_report.md` | Plagiarism check results and any flagged sections |

The `intermediate_data` directory will contain intermediate JSON files that enable
pipeline resume capability:

| File | Description |
|------|-------------|
| `step1_chunks.json` | Volume arcs and semantic chunks per source novel |
| `step2_extracted_plots.json` | Extracted plot atoms (objective + subjective elements) |

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `pipeline/step1_chunking.py` | Semantic chunking & arc anchoring |
| 2 | `pipeline/step2_extraction.py` | Dual-stage plot extraction |
| 3 | `pipeline/step3_knowledge_base.py` | RAG knowledge base & world building |
| 4 | `pipeline/step4_role_casting.py` | Skeleton extraction & role casting |
| 5 | `pipeline/step5_reassembly.py` | Character-driven plot reassembly |
| 6 | `pipeline/step6_generation.py` | Sliding window volume generation |
| 7 | `pipeline/step7_validation.py` | Adversarial plagiarism check & output |

## Configuration Reference

See `config.yaml` for all available settings. All values can also be overridden via environment variables.

Key paths:

| Config key | Env var | Default | Description |
|------------|---------|---------|-------------|
| `paths.input_dir` | `INPUT_DIR` | `./input` | Source novel `.txt` files |
| `paths.output_dir` | `OUTPUT_DIR` | `./output` | Final generated outlines |
| `paths.intermediate_dir` | `INTERMEDIATE_DIR` | `./intermediate_data` | Intermediate JSON outputs for resume |
| `chromadb.path` | `CHROMADB_PATH` | `./chroma_data` | Local ChromaDB vector store |
