from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from pipeline.common.jsonio import read_json, write_json, write_jsonl
from pipeline.common.model import ModelSettings
from pipeline.contracts import (
    BoundaryFrame,
    ChapterSummary,
    ChapterSynopsis,
    ChapterSynopsisBundle,
    HierarchicalOutlineBundle,
    LocalPlotSegment,
    LocalSynopsis,
    OutlineNode,
    SynopsisQuality,
)
from pipeline.modules.module_02_hierarchical_outline.api import prepare_input
from pipeline.modules.module_02_hierarchical_outline.aggregation import story_arc_id
from pipeline.modules.module_02_hierarchical_outline.artifacts import (
    write_outline_artifacts,
)
from pipeline.modules.module_02_hierarchical_outline.input_loader import (
    SynopsisInputError,
    load_accepted_synopsis_input,
)
from pipeline.modules.module_02_hierarchical_outline.parser import (
    OutlineModelContractError,
    parse_boundary_decisions,
    parse_boundary_issues,
    parse_parent_boundary_decisions,
    parse_parent_boundary_issues,
)
from pipeline.modules.module_02_hierarchical_outline.segmentation import (
    apply_boundary_issues,
    story_arc_spans,
)
from pipeline.modules.module_02_hierarchical_outline.service import (
    build_hierarchical_outline,
    build_story_arc_outline,
)
from pipeline.modules.module_02_hierarchical_outline.validation import (
    assess_outline_structure,
)
from pipeline.orchestrator import StageContext


def _synopsis_bundle(
    chapter_id: str, source_hash: str, summaries: tuple[str, ...] = ("人物推进目标。",)
) -> ChapterSynopsisBundle:
    local_synopses = []
    segment_ids = []
    source_ids = []
    for index, summary in enumerate(summaries):
        source_id = f"{chapter_id}:unit:{index}"
        segment_id = f"{chapter_id}:segment:{index}"
        source_ids.append(source_id)
        segment_ids.append(segment_id)
        local_synopses.append(LocalSynopsis(
            window_id=f"{chapter_id}:window:{index}",
            chapter_id=chapter_id,
            reviewed_source_unit_ids=(source_id,),
            segments=(LocalPlotSegment(segment_id, 0, summary, (source_id,)),),
        ))
    synopsis = ChapterSynopsis(
        chapter_id=chapter_id,
        opening_state=BoundaryFrame("人物尚未行动。", (source_ids[0],)),
        chapter_summary=ChapterSummary("随后".join(summaries), tuple(segment_ids)),
        ending_state=BoundaryFrame("人物完成当前推进。", (source_ids[-1],)),
    )
    return ChapterSynopsisBundle(
        chapter_id=chapter_id,
        source_hash=source_hash,
        local_synopses=tuple(local_synopses),
        chapter_synopsis=synopsis,
        quality=SynopsisQuality(
            passed=True,
            content_status="verified",
            source_window_coverage=1.0,
            local_synopsis_count=len(local_synopses),
            local_segment_count=len(segment_ids),
        ),
    )


def _outline_node(
    outline_id: str,
    order: int,
    chapter_ids: tuple[str, ...],
    *,
    level: str = "story_arc",
    children: tuple[str, ...] = (),
) -> OutlineNode:
    return OutlineNode(
        outline_id=outline_id,
        level=level,
        order=order,
        title=f"大纲 {order}",
        chapter_ids=chapter_ids,
        child_outline_ids=children,
        summary="人物在压力下推进目标。",
        opening_situation="人物面对尚未解决的问题。",
        central_goal="解决当前阶段问题。",
        central_conflict="人物行动受到阻力。",
        causal_chain=("阻力促使人物采取行动。",),
        turning_points=("人物改变策略。",),
        ending_change="阶段局面发生变化。",
        open_threads=("后续结果仍未确定。",),
    )


def _node_payload(summary: str = "人物在压力下推进目标并形成阶段结果。") -> dict:
    return {
        "title": "阶段推进",
        "summary": summary,
        "opening_situation": "人物面对尚未解决的问题。",
        "central_goal": "解决当前阶段问题。",
        "central_conflict": "人物行动受到阻力。",
        "causal_chain": ["阻力促使人物采取行动。"],
        "turning_points": ["人物改变策略。"],
        "ending_change": "阶段局面发生变化。",
        "open_threads": ["后续结果仍未确定。"],
        "character_arcs": [],
    }


class FakeCompletion:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, system: str, prompt: str, settings, **kwargs) -> dict:
        self.calls.append({
            "system": system,
            "prompt": prompt,
            "model": settings.model,
            "kwargs": kwargs,
        })
        if not self.responses:
            raise AssertionError("fake completion received an unexpected call")
        return self.responses.pop(0)


def _write_synopsis_input(
    root: Path,
    *,
    accepted: bool = True,
    profile: str = "short_validation",
    complete_work: bool = False,
) -> tuple[Path, Path, tuple[ChapterSynopsisBundle, ...]]:
    stage = root / "module_01_local_synopsis"
    bundles_path = stage / "output" / "chapter_synopsis_bundles.jsonl"
    manifest_path = stage / "manifest.json"
    bundles = (
        _synopsis_bundle("Writer/work-001/0001", "source-hash-1", ("推进一。", "推进二。")),
        _synopsis_bundle("Writer/work-001/0002", "source-hash-2"),
    )
    write_jsonl(bundles_path, [bundle.to_dict() for bundle in bundles])
    chapter_ids = [bundle.chapter_id for bundle in bundles]
    write_json(manifest_path, {
        "module": "module_01_local_synopsis",
        "schema_version": "5.0",
        "author_id": "Writer",
        "work_id": "work-001",
        "profile": profile,
        "accepted": accepted,
        "chapter_ids": chapter_ids,
        "successful_chapter_ids": chapter_ids,
        "failed_chapter_ids": [],
        "input_hashes": {
            bundle.chapter_id: bundle.source_hash for bundle in bundles
        },
        "outputs": {
            "bundles": "output/chapter_synopsis_bundles.jsonl",
        },
        "scope": {
            "available_narrative_chapter_count": len(chapter_ids),
            "selected_narrative_chapter_count": len(chapter_ids),
            "complete_work": complete_work,
        },
    })
    return manifest_path, bundles_path, bundles


class Module02InputFoundationTests(unittest.TestCase):
    def _write_input(
        self, root: Path, *, accepted: bool = True
    ) -> tuple[Path, Path, tuple[ChapterSynopsisBundle, ...]]:
        return _write_synopsis_input(root, accepted=accepted)

    def test_loader_accepts_only_declared_verified_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path, bundles_path, bundles = self._write_input(Path(temp_dir))
            result = load_accepted_synopsis_input(
                manifest_path,
                bundles_path,
                author_id="Writer",
                work_id="work-001",
                profile="short_validation",
            )
        self.assertEqual(result.chapter_ids, tuple(bundle.chapter_id for bundle in bundles))
        self.assertEqual(len(result.synopsis_hashes), 2)
        self.assertTrue(all(len(value) == 64 for value in result.synopsis_hashes))
        self.assertEqual(
            [segment.order for segment in result.chapter_cards[0].local_segments],
            [0, 1],
        )

    def test_loader_rejects_unaccepted_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path, bundles_path, _ = self._write_input(
                Path(temp_dir), accepted=False
            )
            with self.assertRaisesRegex(SynopsisInputError, "has not been accepted"):
                load_accepted_synopsis_input(
                    manifest_path,
                    bundles_path,
                    author_id="Writer",
                    work_id="work-001",
                    profile="short_validation",
                )

    def test_loader_rejects_a_bundle_path_not_declared_by_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path, _, bundles = self._write_input(root)
            other_path = root / "other" / "bundles.jsonl"
            write_jsonl(other_path, [bundle.to_dict() for bundle in bundles])
            with self.assertRaisesRegex(SynopsisInputError, "does not match"):
                load_accepted_synopsis_input(
                    manifest_path,
                    other_path,
                    author_id="Writer",
                    work_id="work-001",
                    profile="short_validation",
                )

    def test_full_book_loader_requires_explicit_narrative_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path, bundles_path, _ = _write_synopsis_input(
                root, profile="full_book", complete_work=True
            )
            manifest = read_json(manifest_path)
            del manifest["scope"]
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(SynopsisInputError, "narrative scope"):
                load_accepted_synopsis_input(
                    manifest_path,
                    bundles_path,
                    author_id="Writer",
                    work_id="work-001",
                    profile="full_book",
                )

    def test_loader_rejects_a_false_complete_work_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path, bundles_path, _ = _write_synopsis_input(
                root, profile="full_book", complete_work=True
            )
            manifest = read_json(manifest_path)
            manifest["scope"]["available_narrative_chapter_count"] += 1
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(SynopsisInputError, "complete_work conflicts"):
                load_accepted_synopsis_input(
                    manifest_path,
                    bundles_path,
                    author_id="Writer",
                    work_id="work-001",
                    profile="full_book",
                )

    def test_stage_context_requires_explicit_upstream_paths(self) -> None:
        context = StageContext(
            author_id="Writer",
            work_id="work-001",
            run_id="test",
            profile="short_validation",
            workspace_root=Path.cwd(),
        )
        with self.assertRaisesRegex(ValueError, "explicit input_paths"):
            prepare_input(context)


class Module02ContractFoundationTests(unittest.TestCase):
    def _valid_bundle(self) -> HierarchicalOutlineBundle:
        chapters = ("Writer/work-001/0001", "Writer/work-001/0002")
        nodes = (
            _outline_node("arc-1", 0, (chapters[0],)),
            _outline_node("arc-2", 1, (chapters[1],)),
        )
        return HierarchicalOutlineBundle(
            author_id="Writer",
            work_id="work-001",
            profile="short_validation",
            source_chapter_ids=chapters,
            source_synopsis_hashes=("synopsis-hash-1", "synopsis-hash-2"),
            nodes=nodes,
            root_outline_ids=("arc-1", "arc-2"),
            aggregation_ceiling="story_arc",
        )

    def test_valid_story_arc_roots_cover_source_in_order(self) -> None:
        bundle = self._valid_bundle()
        bundle.validate()
        self.assertTrue(assess_outline_structure(bundle).passed)

    def test_contract_rejects_an_unknown_schema(self) -> None:
        bundle = replace(self._valid_bundle(), schema_version="9.9")
        with self.assertRaisesRegex(ValueError, "schema version"):
            bundle.validate()

    def test_contract_rejects_misaligned_synopsis_hashes(self) -> None:
        bundle = replace(self._valid_bundle(), source_synopsis_hashes=("only-one",))
        with self.assertRaisesRegex(ValueError, "hashes must align"):
            bundle.validate()

    def test_contract_rejects_incomplete_root_coverage(self) -> None:
        bundle = replace(self._valid_bundle(), root_outline_ids=("arc-1",))
        with self.assertRaisesRegex(ValueError, "cover every source chapter"):
            bundle.validate()

    def test_valid_full_book_hierarchy_uses_each_level(self) -> None:
        base = self._valid_bundle()
        volume = _outline_node(
            "volume-1",
            0,
            base.source_chapter_ids,
            level="volume",
            children=("arc-1", "arc-2"),
        )
        book = _outline_node(
            "book-1",
            0,
            base.source_chapter_ids,
            level="book",
            children=("volume-1",),
        )
        complete = replace(
            base,
            profile="full_book",
            nodes=(*base.nodes, volume, book),
            root_outline_ids=("book-1",),
            aggregation_ceiling="book",
        )
        complete.validate()
        self.assertTrue(assess_outline_structure(complete).passed)

    def test_contract_rejects_a_wrong_parent_child_level(self) -> None:
        base = self._valid_bundle()
        volume = _outline_node(
            "volume-1",
            0,
            base.source_chapter_ids,
            level="volume",
            children=("arc-1", "arc-2"),
        )
        book = _outline_node(
            "book-1",
            0,
            base.source_chapter_ids,
            level="book",
            children=("arc-1",),
        )
        invalid = replace(
            base,
            profile="full_book",
            nodes=(*base.nodes, volume, book),
            root_outline_ids=("book-1",),
            aggregation_ceiling="book",
        )
        report = assess_outline_structure(invalid)
        self.assertFalse(report.passed)
        self.assertIn("must be volume nodes", report.issues[0].message)

    def test_contract_requires_exactly_one_book_root(self) -> None:
        base = self._valid_bundle()
        volume = _outline_node(
            "volume-1",
            0,
            base.source_chapter_ids,
            level="volume",
            children=("arc-1", "arc-2"),
        )
        books = (
            _outline_node(
                "book-1",
                0,
                base.source_chapter_ids,
                level="book",
                children=("volume-1",),
            ),
            _outline_node(
                "book-2",
                1,
                base.source_chapter_ids,
                level="book",
                children=("volume-1",),
            ),
        )
        invalid = replace(
            base,
            profile="full_book",
            nodes=(*base.nodes, volume, *books),
            root_outline_ids=("book-1", "book-2"),
            aggregation_ceiling="book",
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            invalid.validate()


class Module02StoryArcFlowTests(unittest.TestCase):
    def _accepted_input(self, root: Path):
        manifest_path, bundles_path, _ = _write_synopsis_input(root)
        return load_accepted_synopsis_input(
            manifest_path,
            bundles_path,
            author_id="Writer",
            work_id="work-001",
            profile="short_validation",
        )

    def test_boundary_contract_and_program_owned_spans(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_set = self._accepted_input(Path(temp_dir))
        chapter_ids = input_set.chapter_ids
        decisions = parse_boundary_decisions({
            "boundaries": [{
                "left_chapter_id": chapter_ids[0],
                "right_chapter_id": chapter_ids[1],
                "decision": "split",
                "reason": "新的阶段目标开始。",
            }],
        }, chapter_ids)
        spans = story_arc_spans(input_set.chapter_cards, decisions)
        self.assertEqual([span.chapter_ids for span in spans], [
            (chapter_ids[0],),
            (chapter_ids[1],),
        ])

    def test_boundary_review_can_only_flip_a_matching_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_set = self._accepted_input(Path(temp_dir))
        chapter_ids = input_set.chapter_ids
        decisions = parse_boundary_decisions({
            "boundaries": [{
                "left_chapter_id": chapter_ids[0],
                "right_chapter_id": chapter_ids[1],
                "decision": "split",
                "reason": "候选认为阶段变化。",
            }],
        }, chapter_ids)
        issues = parse_boundary_issues({
            "issues": [{
                "left_chapter_id": chapter_ids[0],
                "right_chapter_id": chapter_ids[1],
                "problem": "false_split",
                "reason": "中心目标仍在直接延续。",
            }],
        }, decisions)
        repaired = apply_boundary_issues(decisions, issues)
        self.assertEqual(repaired[0].decision, "continue")
        with self.assertRaisesRegex(OutlineModelContractError, "does not match"):
            parse_boundary_issues({
                "issues": [{
                    "left_chapter_id": chapter_ids[0],
                    "right_chapter_id": chapter_ids[1],
                    "problem": "missed_split",
                    "reason": "不符合当前候选。",
                }],
            }, decisions)

    def test_short_validation_builds_and_writes_a_reviewed_story_arc(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_set = self._accepted_input(root)
            chapter_ids = input_set.chapter_ids
            completion = FakeCompletion([
                {"boundaries": [{
                    "left_chapter_id": chapter_ids[0],
                    "right_chapter_id": chapter_ids[1],
                    "decision": "continue",
                    "reason": "同一阶段目标连续推进。",
                }]},
                {"issues": []},
                _node_payload(),
                {"issues": []},
                {"issues": []},
                {"issues": []},
            ])
            result = build_story_arc_outline(
                input_set,
                settings=ModelSettings(api_key="test-key", model="fake-model"),
                checkpoint_dir=root / "module02" / "checkpoints",
                resume=False,
                completion=completion,
            )
            self.assertTrue(result.accepted, result.failure)
            self.assertEqual(len(result.bundle.nodes), 1)
            self.assertEqual(result.bundle.nodes[0].chapter_ids, chapter_ids)
            self.assertEqual(len(completion.calls), 6)
            self.assertTrue(all(
                "reviewed_source_unit_ids" not in call["prompt"]
                and "source-hash" not in call["prompt"]
                for call in completion.calls
            ))

            paths = write_outline_artifacts(
                root / "module02",
                input_set,
                result,
                run_id="test-run",
                model_policy={"generation_model": "fake-model"},
            )
            manifest = read_json(paths["manifest"])
            self.assertTrue(manifest["accepted"])
            self.assertTrue(paths["outline"].is_file())

    def test_node_is_repaired_once_and_rechecked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_set = self._accepted_input(root)
            chapter_ids = input_set.chapter_ids
            completion = FakeCompletion([
                {"boundaries": [{
                    "left_chapter_id": chapter_ids[0],
                    "right_chapter_id": chapter_ids[1],
                    "decision": "continue",
                    "reason": "阶段连续。",
                }]},
                {"issues": []},
                _node_payload("候选包含一个不受支持的结果。"),
                {"issues": [{
                    "code": "unsupported_claim",
                    "field": "summary",
                    "message": "候选结果超出章纲支持。",
                    "chapter_ids": [chapter_ids[0]],
                    "segment_ids": [],
                }]},
                {"issues": []},
                _node_payload("人物在压力下推进目标。"),
                {"issues": []},
                {"issues": []},
                {"issues": []},
            ])
            result = build_story_arc_outline(
                input_set,
                settings=ModelSettings(api_key="test-key", model="fake-model"),
                checkpoint_dir=root / "module02" / "checkpoints",
                resume=False,
                completion=completion,
            )
        self.assertTrue(result.accepted, result.failure)
        self.assertEqual(len(result.candidate_nodes), 2)
        self.assertEqual(result.bundle.nodes[0].summary, "人物在压力下推进目标。")
        self.assertEqual(len(completion.calls), 9)

    def test_top_down_issue_rejects_the_formal_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_set = self._accepted_input(root)
            chapter_ids = input_set.chapter_ids
            completion = FakeCompletion([
                {"boundaries": [{
                    "left_chapter_id": chapter_ids[0],
                    "right_chapter_id": chapter_ids[1],
                    "decision": "continue",
                    "reason": "阶段连续。",
                }]},
                {"issues": []},
                _node_payload(),
                {"issues": []},
                {"issues": []},
                {"issues": [{
                    "code": "invalid_boundary",
                    "field": "boundary",
                    "message": "整个样本显示这里存在错误的阶段合并。",
                    "chapter_ids": list(chapter_ids),
                    "segment_ids": [],
                }]},
            ])
            result = build_story_arc_outline(
                input_set,
                settings=ModelSettings(api_key="test-key", model="fake-model"),
                checkpoint_dir=root / "module02" / "checkpoints",
                resume=False,
                completion=completion,
            )
            paths = write_outline_artifacts(
                root / "module02",
                input_set,
                result,
                run_id="test-run",
                model_policy={"generation_model": "fake-model"},
            )
            manifest = read_json(paths["manifest"])
        self.assertFalse(result.accepted)
        self.assertEqual(result.failure["code"], "OUTLINE_SEMANTIC_REVIEW_FAILED")
        self.assertFalse(manifest["accepted"])


class Module02ParentAggregationTests(unittest.TestCase):
    def _accepted_input(
        self,
        root: Path,
        *,
        profile: str = "short_validation",
        complete_work: bool = False,
    ):
        manifest_path, bundles_path, _ = _write_synopsis_input(
            root,
            profile=profile,
            complete_work=complete_work,
        )
        return load_accepted_synopsis_input(
            manifest_path,
            bundles_path,
            author_id="Writer",
            work_id="work-001",
            profile=profile,
        )

    def test_parent_boundary_contract_preserves_story_arc_order(self) -> None:
        outline_ids = ("arc-1", "arc-2")
        decisions = parse_parent_boundary_decisions({
            "boundaries": [{
                "left_outline_id": "arc-1",
                "right_outline_id": "arc-2",
                "decision": "split",
                "reason": "宏观阶段已转换。",
            }],
        }, outline_ids)
        issues = parse_parent_boundary_issues({
            "issues": [{
                "left_outline_id": "arc-1",
                "right_outline_id": "arc-2",
                "problem": "false_split",
                "reason": "长期目标仍在连续推进。",
            }],
        }, decisions)
        self.assertEqual(decisions[0].decision, "split")
        self.assertEqual(issues[0].problem, "false_split")

        with self.assertRaisesRegex(OutlineModelContractError, "preserve"):
            parse_parent_boundary_decisions({
                "boundaries": [{
                    "left_outline_id": "arc-2",
                    "right_outline_id": "arc-1",
                    "decision": "continue",
                    "reason": "顺序错误。",
                }],
            }, outline_ids)

    def test_volume_aggregates_reviewed_story_arcs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_set = self._accepted_input(root)
            chapter_ids = input_set.chapter_ids
            arc_ids = tuple(
                story_arc_id(input_set.work_id, (chapter_id,))
                for chapter_id in chapter_ids
            )
            completion = FakeCompletion([
                {"boundaries": [{
                    "left_chapter_id": chapter_ids[0],
                    "right_chapter_id": chapter_ids[1],
                    "decision": "split",
                    "reason": "右章开始新的阶段目标。",
                }]},
                {"issues": []},
                _node_payload("第一故事弧形成阶段变化。"),
                {"issues": []},
                {"issues": []},
                _node_payload("第二故事弧继续宏观目标。"),
                {"issues": []},
                {"issues": []},
                {"issues": []},
                {"boundaries": [{
                    "left_outline_id": arc_ids[0],
                    "right_outline_id": arc_ids[1],
                    "decision": "continue",
                    "reason": "两个故事弧属于同一宏观阶段。",
                }]},
                {"issues": []},
                _node_payload("两个故事弧共同构成一个语义卷。"),
                {"issues": []},
                {"issues": []},
                {"issues": []},
            ])
            result = build_hierarchical_outline(
                input_set,
                aggregation_ceiling="volume",
                settings=ModelSettings(api_key="test-key", model="fake-model"),
                checkpoint_dir=root / "module02" / "checkpoints",
                resume=False,
                completion=completion,
            )
            self.assertTrue(result.accepted, result.failure)
            self.assertEqual(
                [node.level for node in result.bundle.nodes],
                ["story_arc", "story_arc", "volume"],
            )
            volume = result.bundle.nodes[-1]
            self.assertEqual(volume.child_outline_ids, arc_ids)
            self.assertEqual(volume.chapter_ids, chapter_ids)
            self.assertEqual(result.bundle.root_outline_ids, (volume.outline_id,))
            self.assertEqual(len(result.volume_boundary_decisions), 1)
            self.assertEqual(len(completion.calls), 15)

            paths = write_outline_artifacts(
                root / "module02",
                input_set,
                result,
                run_id="volume-run",
                model_policy={"generation_model": "fake-model"},
            )
            quality = read_json(paths["quality"])
            manifest = read_json(paths["manifest"])
        self.assertEqual(quality["story_arc_count"], 2)
        self.assertEqual(quality["volume_count"], 1)
        self.assertEqual(quality["book_count"], 0)
        self.assertEqual(manifest["aggregation_ceiling"], "volume")

    def test_book_requires_confirmed_complete_work_before_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_set = self._accepted_input(root)
            completion = FakeCompletion([])
            result = build_hierarchical_outline(
                input_set,
                aggregation_ceiling="book",
                settings=ModelSettings(api_key="test-key", model="fake-model"),
                checkpoint_dir=root / "module02" / "checkpoints",
                resume=False,
                completion=completion,
            )
        self.assertFalse(result.accepted)
        self.assertEqual(result.failure["code"], "BOOK_REQUIRES_COMPLETE_WORK")
        self.assertEqual(completion.calls, [])

    def test_book_is_the_unique_root_of_a_complete_hierarchy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_set = self._accepted_input(
                root, profile="full_book", complete_work=True
            )
            chapter_ids = input_set.chapter_ids
            completion = FakeCompletion([
                {"boundaries": [{
                    "left_chapter_id": chapter_ids[0],
                    "right_chapter_id": chapter_ids[1],
                    "decision": "continue",
                    "reason": "章节属于同一故事弧。",
                }]},
                {"issues": []},
                _node_payload("完整故事弧。"),
                {"issues": []},
                {"issues": []},
                {"issues": []},
                _node_payload("完整语义卷。"),
                {"issues": []},
                {"issues": []},
                {"issues": []},
                _node_payload("完整作品级梗概。"),
                {"issues": []},
                {"issues": []},
                {"issues": []},
            ])
            result = build_hierarchical_outline(
                input_set,
                aggregation_ceiling="book",
                settings=ModelSettings(api_key="test-key", model="fake-model"),
                checkpoint_dir=root / "module02" / "checkpoints",
                resume=False,
                completion=completion,
            )
        self.assertTrue(result.accepted, result.failure)
        self.assertEqual(
            [node.level for node in result.bundle.nodes],
            ["story_arc", "volume", "book"],
        )
        book = result.bundle.nodes[-1]
        volume = result.bundle.nodes[-2]
        self.assertEqual(book.order, 0)
        self.assertEqual(book.child_outline_ids, (volume.outline_id,))
        self.assertEqual(result.bundle.root_outline_ids, (book.outline_id,))
        self.assertEqual(result.bundle.aggregation_ceiling, "book")
        self.assertEqual(len(completion.calls), 14)


if __name__ == "__main__":
    unittest.main()
