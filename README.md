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
   - Set `input_dir` (default `./input`) and `output_dir` (default `./output`).

3. **Add source novels**
   - Place your source novel `.txt` files in the `input` directory.

4. **Run**
   ```bash
   python main.py
   ```

## Output

After a successful run, the `output` directory will contain:

| File | Description |
|------|-------------|
| `novel_world_bible.md` | New cultivation system, protagonist sheet, character relationships |
| `volume_1_to_N_outline.md` | Full per-chapter outline split by cultivation arc |
| `validation_report.md` | Plagiarism check results and any flagged sections |

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
