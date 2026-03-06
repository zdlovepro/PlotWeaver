"""
PlotWeaver Pipeline Orchestrator
Runs all 15 steps to generate a new novel outline from multiple input novels.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import yaml  # type: ignore[import]

from .llm import create_client, PromptManager
from .preprocess import load_novel, process_novels_parallel
from .extraction import PlotExtractor
from .analysis import NarrativeAnnotator, EmotionAnalyzer, TensionCurve, CausalGraph
from .rag import Embedder, VectorStore, MultiRetriever
from .remix import CharacterMapper, CompatibilityScorer, ReplacementEngine, ConsistencyChecker
from .generation import OutlineGenerator, OutlineFormatter
from .models import Novel, Character, AnnotatedPlotUnit, CharacterMapping


class Pipeline:
    """Orchestrates all 15 steps of the PlotWeaver pipeline."""

    def __init__(self, config_path: str = "config/pipeline.yaml"):
        self.config = self._load_config(config_path)
        self._setup_directories()

        # LLM client
        llm_cfg = self.config.get("llm", {})
        self.llm = create_client(
            provider=llm_cfg.get("provider", "ollama"),
            api_key=llm_cfg.get("api_key", ""),
            model=llm_cfg.get("model", "qwen2.5"),
            host=llm_cfg.get("ollama_host", "http://localhost:11434"),
            temperature=float(llm_cfg.get("temperature", 0.7)),
        )
        self.prompts = PromptManager(
            prompts_dir=self.config.get("pipeline", {}).get("prompts_dir")
        )

        # Embedding
        emb_cfg = self.config.get("embedding", {})
        self.embedder = Embedder(
            provider=emb_cfg.get("provider", "local"),
            model=emb_cfg.get("model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            api_key=emb_cfg.get("api_key", ""),
        )

        # RAG
        rag_cfg = self.config.get("rag", {})
        self.vector_store = VectorStore(
            persist_directory=rag_cfg.get("chromadb_path", "data/vectordb")
        )
        self.retriever = MultiRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=int(rag_cfg.get("top_k", 10)),
        )

        # Pipeline config
        pipe_cfg = self.config.get("pipeline", {})
        self.framework_novel_id = pipe_cfg.get("framework_novel", None)
        self.max_replacement_depth = int(pipe_cfg.get("replacement_depth", 3))
        self.parallel_workers = int(pipe_cfg.get("parallel_workers", 4))
        self.intermediate_dir = Path("data/intermediate")
        self.output_dir = Path("data/output")

    def _load_config(self, config_path: str) -> dict:
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _setup_directories(self) -> None:
        for d in ["data/intermediate", "data/vectordb", "data/output"]:
            os.makedirs(d, exist_ok=True)

    def _save_json(self, path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_json(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _log(self, step: int, message: str) -> None:
        try:
            from rich.console import Console  # type: ignore[import]
            console = Console()
            console.print(f"[bold cyan]Step {step:02d}[/bold cyan] {message}")
        except ImportError:
            print(f"[Step {step:02d}] {message}")

    # -------------------------------------------------------------------------
    # Phase 1 & 2: Preprocessing (Steps 1–2 per novel)
    # -------------------------------------------------------------------------

    def run_preprocess(self, novel_files: list[str]) -> list[Novel]:
        """Steps 1–2: Read, clean, and chapter-split all input novels."""
        self._log(1, f"预处理 {len(novel_files)} 部小说...")
        novels = process_novels_parallel(novel_files, max_workers=self.parallel_workers)

        for novel in novels:
            out_path = self.intermediate_dir / novel.id / "chapters.json"
            self._save_json(out_path, novel.to_dict())
            self._log(2, f"保存章节: {out_path}  ({len(novel.chapters)} 章)")

        return novels

    # -------------------------------------------------------------------------
    # Phase 2 (extraction): Step 3
    # -------------------------------------------------------------------------

    def run_extraction(self, novels: list[Novel]) -> dict[str, list[AnnotatedPlotUnit]]:
        """Step 3: Extract plot units from all novels using LLM."""
        self._log(3, "提取情节单元 (LLM)...")
        extractor = PlotExtractor(self.llm, self.prompts)
        novel_plots: dict[str, list] = {}

        for novel in novels:
            self._log(3, f"  处理 [{novel.id}] {len(novel.chapters)} 章...")
            units = extractor.extract_from_novel(novel.chapters)
            novel_plots[novel.id] = units
            out_path = self.intermediate_dir / novel.id / "raw_plots.json"
            self._save_json(out_path, [u.to_dict() for u in units])
            self._log(3, f"  → {len(units)} 个情节单元")

        return novel_plots

    # -------------------------------------------------------------------------
    # Phase 3 (analysis): Steps 4–7
    # -------------------------------------------------------------------------

    def run_analysis(self, novel_plots: dict[str, list]) -> dict[str, list[AnnotatedPlotUnit]]:
        """Steps 4–7: Annotate, analyze emotion/tension/causality."""
        annotator = NarrativeAnnotator(self.llm, self.prompts)
        emotion_analyzer = EmotionAnalyzer()
        tension_analyzer = TensionCurve()
        causal_builder = CausalGraph()

        annotated_novels: dict[str, list[AnnotatedPlotUnit]] = {}

        for novel_id, units in novel_plots.items():
            # Step 4: LLM annotation
            self._log(4, f"叙事标注 [{novel_id}] (LLM)...")
            annotated = annotator.annotate_all(units)
            out4 = self.intermediate_dir / novel_id / "annotated_plots.json"
            self._save_json(out4, [u.to_dict() for u in annotated])

            # Step 5: Emotion curve
            self._log(5, f"情感曲线分析 [{novel_id}]...")
            emotion_curve = emotion_analyzer.build_emotion_curve(annotated)
            out5 = self.intermediate_dir / novel_id / "emotion_curve.json"
            self._save_json(out5, emotion_curve)

            # Step 6: Tension curve
            self._log(6, f"张力曲线分析 [{novel_id}]...")
            tension_curve = tension_analyzer.build_tension_curve(annotated)
            out6 = self.intermediate_dir / novel_id / "tension_curve.json"
            self._save_json(out6, tension_curve)

            # Step 7: Causal graph
            self._log(7, f"因果图构建 [{novel_id}]...")
            causal = causal_builder.build(annotated)
            out7 = self.intermediate_dir / novel_id / "causal_graph.json"
            self._save_json(out7, causal)

            annotated_novels[novel_id] = annotated

        return annotated_novels

    # -------------------------------------------------------------------------
    # Phase 4 (RAG): Steps 8–9
    # -------------------------------------------------------------------------

    def run_rag_build(self, annotated_novels: dict[str, list[AnnotatedPlotUnit]]) -> None:
        """Steps 8–9: Build vector knowledge base."""
        self._log(8, "生成嵌入向量...")
        self._log(9, "构建 ChromaDB 知识库...")

        for novel_id, units in annotated_novels.items():
            texts = [u.plot_unit.content for u in units]
            embeddings = self.embedder.embed(texts)
            self.vector_store.add_plot_units(units, embeddings)
            self._log(9, f"  已存储 [{novel_id}] {len(units)} 个情节单元")

        self._log(9, f"知识库总记录数: {self.vector_store.count_plots()}")

    # -------------------------------------------------------------------------
    # Phase 5 (character mapping): Step 10
    # -------------------------------------------------------------------------

    def run_character_mapping(
        self,
        framework_novel_id: str,
        annotated_novels: dict[str, list[AnnotatedPlotUnit]],
    ) -> list[CharacterMapping]:
        """Step 10: Map characters between framework and donor novels."""
        self._log(10, "人物映射 (匈牙利算法)...")

        def _extract_characters(novel_id: str, units: list[AnnotatedPlotUnit]) -> list[Character]:
            char_set: dict[str, Character] = {}
            for u in units:
                for name in u.plot_unit.characters:
                    if name not in char_set:
                        char_set[name] = Character(
                            id=f"{novel_id}_{name}",
                            novel_id=novel_id,
                            name=name,
                        )
            return list(char_set.values())

        mapper = CharacterMapper(self.embedder)
        fw_units = annotated_novels.get(framework_novel_id, [])
        fw_chars = _extract_characters(framework_novel_id, fw_units)

        all_mappings: list[CharacterMapping] = []
        for novel_id, units in annotated_novels.items():
            if novel_id == framework_novel_id:
                continue
            donor_chars = _extract_characters(novel_id, units)
            mappings = mapper.map_characters(fw_chars, donor_chars)
            all_mappings.extend(mappings)

        out10 = self.intermediate_dir / "character_mapping.json"
        self._save_json(out10, [m.to_dict() for m in all_mappings])
        self._log(10, f"→ {len(all_mappings)} 个人物映射")
        return all_mappings

    # -------------------------------------------------------------------------
    # Phase 6 (remix): Steps 11–13
    # -------------------------------------------------------------------------

    def run_remix(
        self,
        framework_novel_id: str,
        annotated_novels: dict[str, list[AnnotatedPlotUnit]],
        character_mappings: list[CharacterMapping],
    ) -> list[AnnotatedPlotUnit]:
        """Steps 11–13: Score compatibility, replace units, validate."""
        scorer = CompatibilityScorer()
        replacement_engine = ReplacementEngine(
            scorer=scorer,
            max_replacement_depth=self.max_replacement_depth,
        )
        checker = ConsistencyChecker()

        fw_units = annotated_novels.get(framework_novel_id, [])

        # Collect all donor units
        donor_units: list[AnnotatedPlotUnit] = []
        for novel_id, units in annotated_novels.items():
            if novel_id != framework_novel_id:
                donor_units.extend(units)

        # Step 11: Compatibility matrix
        self._log(11, "计算兼容性矩阵...")
        if donor_units:
            matrix = scorer.build_compatibility_matrix(fw_units[:20], donor_units[:20])
            out11 = self.intermediate_dir / "compatibility_matrix.json"
            self._save_json(out11, matrix)

        # Step 12: Replacement
        self._log(12, f"情节替换 (贪心搜索, depth={self.max_replacement_depth})...")
        remixed = replacement_engine.remix(fw_units, donor_units)
        out12 = self.intermediate_dir / "remixed_plots.json"
        self._save_json(out12, [u.to_dict() for u in remixed])
        self._log(12, f"→ {len(remixed)} 个混合情节单元")

        # Step 13: Validation
        self._log(13, "一致性验证...")
        report = checker.validate(remixed, character_mappings)
        out13 = self.intermediate_dir / "validation_report.json"
        self._save_json(out13, report)
        status = "✓ 通过" if report["passed"] else f"⚠ 发现 {len(report['issues'])} 个问题"
        self._log(13, f"验证结果: {status}")

        return remixed

    # -------------------------------------------------------------------------
    # Phase 7 (generation): Steps 14–15
    # -------------------------------------------------------------------------

    def run_generation(
        self,
        remixed_units: list[AnnotatedPlotUnit],
        character_mappings: list[CharacterMapping],
        framework_novel_id: str,
    ) -> str:
        """Steps 14–15: Generate outline with LLM and format to Markdown."""
        # Load curves for framework novel
        emotion_curve = None
        tension_curve_data = None
        try:
            emotion_curve = self._load_json(self.intermediate_dir / framework_novel_id / "emotion_curve.json")
        except FileNotFoundError:
            pass
        try:
            tension_curve_data = self._load_json(self.intermediate_dir / framework_novel_id / "tension_curve.json")
        except FileNotFoundError:
            pass

        # Step 14: LLM outline generation
        self._log(14, "生成小说大纲 (LLM)...")
        generator = OutlineGenerator(self.llm, self.prompts)
        outline = generator.generate(remixed_units, character_mappings, emotion_curve, tension_curve_data)

        # Step 15: Format and save
        self._log(15, "格式化输出 Markdown...")
        formatter = OutlineFormatter()
        output_path = str(self.output_dir / "新小说大纲.md")
        formatter.save(outline, output_path)
        self._log(15, f"✓ 已保存: {output_path}")
        return output_path

    # -------------------------------------------------------------------------
    # Full pipeline runner
    # -------------------------------------------------------------------------

    def run(self, novel_files: Optional[list[str]] = None) -> str:
        """Run the complete 15-step pipeline."""
        if novel_files is None:
            novel_files = self._discover_input_files()

        if not novel_files:
            raise ValueError("No input novel files found in data/input/. Please add .txt files.")

        self._log(0, f"PlotWeaver 启动 — 处理 {len(novel_files)} 部小说")
        start_time = time.time()

        # Phase 1–2
        novels = self.run_preprocess(novel_files)

        # Determine framework novel
        fw_id = self.framework_novel_id or novels[0].id

        # Phase 2 (extraction)
        novel_plots = self.run_extraction(novels)

        # Phase 3 (analysis)
        annotated_novels = self.run_analysis(novel_plots)

        # Phase 4 (RAG)
        self.run_rag_build(annotated_novels)

        # Phase 5 (character mapping)
        character_mappings = self.run_character_mapping(fw_id, annotated_novels)

        # Phase 6 (remix)
        remixed = self.run_remix(fw_id, annotated_novels, character_mappings)

        # Phase 7 (generation)
        output_path = self.run_generation(remixed, character_mappings, fw_id)

        elapsed = time.time() - start_time
        self._log(0, f"完成！耗时 {elapsed:.1f}s  输出: {output_path}")
        return output_path

    def _discover_input_files(self) -> list[str]:
        """Scan data/input/ for .txt files."""
        input_dir = Path("data/input")
        if not input_dir.is_dir():
            return []
        return [str(p) for p in sorted(input_dir.glob("*.txt"))]
