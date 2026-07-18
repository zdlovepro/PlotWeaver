# PlotWeaver V2.0

An automated Python pipeline for fusing multiple Xianxia (Chinese cultivation)
novel outlines into a new, highly coherent novel outline.

## Features

- **13-step automated pipeline** powered by DeepSeek API and ChromaDB
- Detailed architecture notes in `docs/pipeline.md`
- Shared non-step logic centralized under `pipeline/core/`
- Semantic arc detection with long-chapter subchunk support
- Dual-stage plot extraction with `raw + canonical + character_keys`
- Event induction that links multiple atoms into larger event units
- Retrieval index plus world/template mining
- Multi-source skeleton casting and separate character-casting pass
- Character-driven plot reassembly
- Sliding-window volume generation
- Validation and final output packaging

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure**
   - Edit `config.yaml` and set your `DEEPSEEK_API_KEY`, or set the environment variable.
   - Set `input_dir`, `output_dir`, and `intermediate_dir` as needed.

3. **Add source novels**
   - Place your source novel `.txt` files in the configured input directory.

4. **Run**
   ```bash
   python main.py
   ```

## Running the Pipeline (`--start-step`, `--end-step`, `--only-step`)

After early steps finish, their outputs are written into `intermediate_data`
and can be reused with `--start-step`. You can also stop after a specific
step with `--end-step`, or run a single step with `--only-step`.

### Common commands

```bash
# Full run
python main.py

# Resume from Step 3
python main.py --start-step 3

# Run only the material-building phase
python main.py --start-step 1 --end-step 4

# Run only Step 2
python main.py --only-step 2

# Run the world/pattern/template phase
python main.py --start-step 5 --end-step 7

# Regenerate only Step 7 templates
python main.py --only-step 7
```

### Resume Safety

Starting from Step 1 creates `intermediate_data/pipeline_run_manifest.json`.
When resuming, PlotWeaver verifies the hashes of prior artifacts so outputs from
different runs are not silently mixed. For a one-time recovery of legacy
intermediate files that predate the manifest, explicitly opt in:

```powershell
$env:PLOTWEAVER_ALLOW_LEGACY_RESUME = "1"
py -3 main.py --start-step 3 --end-step 9
```

The resumed steps are then recorded and verified normally. Remove the
environment variable after the recovery run.

### Phase notes

Steps 1-4 are the source-material accumulation stage:

- Step 1: physical chunking
- Step 2: plot atom extraction
- Step 3: event induction
- Step 4: RAG knowledge base

Steps 5 onward move into synthesis and outline generation:

- world fusion
- pattern/template mining
- skeleton fusion
- outline generation

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `pipeline/step1_chunking.py` | Physical chunking and arc anchoring |
| 2 | `pipeline/step2_extraction.py` | Plot atom extraction |
| 3 | `pipeline/step3_event_induction.py` | Event induction from plot atoms |
| 4 | `pipeline/step4_knowledge_base.py` | Build the RAG knowledge base |
| 5 | `pipeline/step5_world_fusion.py` | Build the fused world base |
| 6 | `pipeline/step6_interaction_mining.py` | Mine reusable interactions and long threads |
| 7 | `pipeline/step7_template_mining.py` | Derive event, volume, and chapter-flow templates |
| 8 | `pipeline/step8_skeleton_extraction.py` | Extract per-source skeleton nodes |
| 9 | `pipeline/step9_skeleton_fusion.py` | Fuse multi-source skeletons into one backbone |
| 10 | `pipeline/step10_character_casting.py` | Cast protagonist, supporting roles, and relationship stages |
| 11 | `pipeline/step11_reassembly.py` | Character-driven plot reassembly |
| 12 | `pipeline/step12_generation.py` | Sliding-window volume generation |
| 13 | `pipeline/step13_validation.py` | Validation and final output |

## Repository Layout

- `pipeline/`: numbered step modules plus `pipeline/core/` shared modules
- `pipeline/core/`: shared data structures, utilities, aliasing, world-building helpers, and skeleton helpers
- `docs/`: pipeline and architecture notes

## Intermediate Outputs

Common intermediate files now include:

| File | Description |
|------|-------------|
| `step1_chunks.json` | Volume arcs and chunked source events |
| `step2_extracted_plots.json` | Extracted plot atoms |
| `step3_induced_events.json` | Multi-atom induced events |
| `step5_world_fusion.json` | Fused world snapshot |
| `step6_interaction_mining.json` | Pattern-enhanced world snapshot |
| `step7_template_mining.json` | Template-mined world snapshot |
| `step8_source_skeletons.json` | Per-source skeleton node maps |
| `step9_skeleton.json` | Fused event skeleton |
| `step10_casted_skeleton.json` | Skeleton with character sheet |
| `step11_reassembled_plot.json` | Reassembled event plan |
| `step12_volume_outlines.json` | Generated volume outlines |

## Output

After a successful run, the `output` directory contains:

- `novel_world_bible.md`
- `volume_1_to_N_outline.md`
- `validation_report.md`
