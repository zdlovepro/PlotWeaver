# PlotWeaver

PlotWeaver is an evidence-first pipeline for distilling a reusable **小说作者 Skill** from Chinese web-novel samples. The author identifier is supplied at runtime; it becomes the output folder name, for example `output/ExampleAuthor_skill/`.

The V3 path never uses source prose as a generation prompt. It first turns each chapter into source-evidenced facts, events, scenes, time/space relations and dramatic beats; only then can it build bounded generation programs and judge reconstruction gaps.

## Project documentation

The current V3/V4 documentation replaces the retired V2 thirteen-step design:

- [Requirements specification](docs/SRS.md)
- [Software design specification](docs/SDS.md)
- [Local models, LoRA, SFT, DPO and style evaluation](docs/MODEL_GUIDE.md)
- [Implementation roadmap](docs/ROADMAP.md)

## V3 numbered pipeline

```text
1. 导入小说文本
2. 高密度、可追溯的章节标注
3. 跨章节连续性图与本地剧情事实图谱 `narrative_graph.json`
4. 匿名化多层叙事模板
5. 量化文风蒸馏
6. 编译小说作者 Skill
7. 可选：受控扩写章节规划（不自动生成正文）
```

Put one or more `.txt` novels in `input/` and run:

```powershell
python main.py run --author-id ExampleAuthor --work-id work-001 --chapter-limit 20
```

`input/` is the active import source. `novels/` can retain the complete source archive but is not read automatically. Importing a directory assigns stable IDs such as `work-001`; original titles remain in metadata. One compiled skill currently corresponds to one selected work, so a multi-work import must specify `--work-id` when running stage 6 or 7. This prevents silently overwriting the author-level output with whichever work happened to finish last.

Every run writes a hash manifest to `runs/<AuthorId>/<RunId>/pipeline_manifest.json`. Resume an interrupted annotation pass with the same run inputs; add `--no-resume` only when the source or extraction contract has changed.

## High-density annotation

The annotation stage uses short overlapping source windows and checkpointed batches, so long chapters are never sent in one request. Every accepted entity, fact, event and scene retains an exact source quote and offset. Hard quality gates require event, fact and scene density proportional to source length, coverage of evidence-bearing sections, ordered time relations, fact-backed spatial relations, and valid state transitions.

```powershell
python main.py ingest --author-id ExampleAuthor --source-dir input
python main.py annotate --author-id ExampleAuthor --work-id work-001 --limit 20
python main.py continuity --author-id ExampleAuthor --work-id work-001 --limit 20
python main.py graph --author-id ExampleAuthor --work-id work-001 --limit 20
python main.py templates --author-id ExampleAuthor --work-id work-001 --limit 20 --batch-size 5
python main.py style --author-id ExampleAuthor --work-id work-001 --limit 20 --batch-size 5
python main.py compile-skill --author-id ExampleAuthor --work-id work-001 --limit 20
```

The first five commands produce typed data under `corpus/<AuthorId>/<WorkId>/`. `compile-skill` validates matching samples and emits a generic package at `output/<AuthorId>_skill/`, including `SKILL.md`, anonymous narrative templates, a measured style profile and Chinese JSON prompt templates.

## Controlled expansion and reconstruction tests

`compile-program` derives an immutable `事实 → 事件 → 场景 → 节拍 → 段落` contract for a chapter. A fact/event is assigned to exactly one local scene and later events are explicit no-leak barriers.

```powershell
python main.py compile-program --author-id ExampleAuthor --work-id work-001 --chapter 1 --limit 20
python main.py plan-expansion --author-id ExampleAuthor --work-id work-001 --chapter 1 --limit 20 --target-char-min 4000 --target-char-max 5000
```

The controlled-expansion planner allocates a hard character budget to scenes and paragraphs. Every added beat is labelled either **合理推导** or **氛围扩写** and links to local supporting fact IDs. Such labels allow actions, reactions, transitions and sensory detail, but never authorize a new relationship, setting, resource, motive, time point or plot result.

Generation is separate from planning so a failed candidate cannot be mistaken for the skill itself:

```powershell
python main.py generate-from-program --author-id ExampleAuthor `
  --program runs/ExampleAuthor/chapter-plan/chapter_programs/chapter-0001.program.json `
  --graph corpus/ExampleAuthor/work-001/narrative_graph.json --run-id first5-test
```

The generator works scene by scene and sends exactly one paragraph job per writing request. Before that request, it reads only the paragraph’s bounded subgraph from `narrative_graph.json`: permitted entities, facts, relationships, causal links, foreshadow lifecycle and previously committed state. It never sends source prose, a whole source chapter, or generated prose tails to the writer. Each chapter first writes `chapter_graph_patch.candidate.json`; semantic, prose, role, style, length and state/causality checks must all pass before the patch is committed, `generated_draft.txt` is published, and the next chapter may start. Checkpoints make model transport failures resumable.

For a planned chapter range, use the sequence runner rather than invoking each chapter independently:

```powershell
python main.py generate-program-work --author-id ExampleAuthor --index runs/ExampleAuthor/chapter-plan-work/expansion_plan_index.json --run-id first5-test
```

It creates an input state file for chapter 2 onward only after the preceding chapter has passed every gate. A rejected chapter blocks later chapters and leaves their status as `not_started`; this prevents a speculative exit state from contaminating continuity evaluation.

After generation, run the independent reconstruction regression:

```powershell
python main.py evaluate-program-fidelity --author-id ExampleAuthor --work-id work-001 `
  --index runs/ExampleAuthor/chapter-plan-work/expansion_plan_index.json `
  --sequence runs/ExampleAuthor/first5-test/program_sequence/sequence_summary.json `
  --annotation-limit 20 --run-id first5-fidelity
```

The report contains a local style-similarity measurement and a strict model judgment over anonymised facts, events, scenes, state changes and time/space relations. Any explicit omission or contradiction forces the structural result to fail even if the model's numeric score is high.

## Run options

```powershell
python main.py run --author-id ExampleAuthor --work-id work-001 `
  --chapter-limit 20 --annotation-input-chars 1200 --annotation-overlap-units 1 `
  --template-batch-size 5 --style-batch-size 5
```

To include planning for the first five chapters after stages 1–6:

```powershell
python main.py run --author-id ExampleAuthor --work-id work-001 --chapter-limit 20 `
  --expansion-chapters 5 --target-char-min 4000 --target-char-max 5000
```

All model-facing structured prompts are Chinese and give a concrete JSON example. Strict JSON validation accepts exactly the declared V3 format; it also recognises the earlier faithful chapter-program schema with its documented defaults, so existing faithful program artifacts remain readable.

Use `--offline` only for deterministic structural smoke tests after annotation has completed; high-density evidence annotation itself requires a model. Configure a compatible model in `config.yaml` for annotation, template mining, style synthesis and generation. Analyse only novels you are permitted to process.
