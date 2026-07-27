from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.pipeline import PipelineOptions, run_pipeline


class V3PipelineTests(unittest.TestCase):
    def test_run_composes_only_v3_stages_and_keeps_generation_out_of_band(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            work = base / "work-001"
            work.mkdir()
            annotation = work / "chapter_annotations.sample-2.jsonl"; annotation.touch()
            annotation_quality = work / "chapter_annotations.sample-2.quality.json"; annotation_quality.touch()
            continuity = work / "work_continuity.sample-2.json"; continuity.touch()
            continuity_quality = work / "work_continuity.sample-2.quality.json"; continuity_quality.touch()
            narrative_graph = work / "narrative_graph.json"; narrative_graph.touch()
            narrative_graph_quality = work / "narrative_graph.quality.json"; narrative_graph_quality.touch()
            templates = work / "narrative_templates.sample-2.json"; templates.touch()
            templates_quality = work / "narrative_templates.sample-2.quality.json"; templates_quality.touch()
            cards = work / "style_cards.sample-2.jsonl"; cards.touch()
            profile = work / "author_style_profile.sample-2.json"; profile.touch()
            style_quality = work / "author_style_profile.sample-2.quality.json"; style_quality.touch()
            skill = base / "Example_skill"; skill.mkdir()
            (skill / "SKILL.md").touch()
            skill_quality = skill / "skill_quality.json"; skill_quality.touch()
            plan_index = base / "expansion_plan_index.json"; plan_index.touch()
            probe = base / "model_probe.json"; probe.touch()

            with patch("pipeline.pipeline.initialize_manifest"), \
                 patch("pipeline.pipeline.record_stage") as record, \
                 patch("pipeline.pipeline.ingest_directory"), \
                 patch("pipeline.pipeline.active_work_ids", return_value=["work-001"]), \
                 patch("pipeline.pipeline.corpus_dir", return_value=work), \
                 patch("pipeline.pipeline.output_dir", return_value=skill), \
                 patch("pipeline.pipeline.probe_model", return_value=probe) as model_probe, \
                 patch("pipeline.pipeline.annotate_work", return_value=(annotation, annotation_quality)) as annotate, \
                 patch("pipeline.pipeline.build_continuity_work", return_value=(continuity, continuity_quality)) as build_continuity, \
                 patch("pipeline.pipeline.build_narrative_graph_work", return_value=(narrative_graph, narrative_graph_quality)) as build_graph, \
                 patch("pipeline.pipeline.mine_template_library", return_value=(templates, templates_quality)) as mine_templates, \
                 patch("pipeline.pipeline.distil_style_profile", return_value=(cards, profile, style_quality)) as distil_style, \
                 patch("pipeline.pipeline.compile_author_skill", return_value=(skill, skill_quality)) as compile_skill, \
                 patch("pipeline.pipeline.plan_expansion_programs_work", return_value=plan_index) as plan_expansion:
                result = run_pipeline(PipelineOptions(
                    author_id="Example", work_id="work-001", chapter_limit=2,
                    expansion_chapters=2, target_char_min=4_000, target_char_max=5_000,
                ))

            self.assertEqual(result, skill)
            model_probe.assert_called_once_with("Example", run_id="pipeline-annotation-probe")
            annotate.assert_called_once()
            build_continuity.assert_called_once()
            build_graph.assert_called_once_with("Example", "work-001", limit=2)
            mine_templates.assert_called_once()
            distil_style.assert_called_once()
            compile_skill.assert_called_once_with("Example", "work-001", limit=2)
            plan_expansion.assert_called_once_with(
                "Example", "work-001", chapter_limit=2, target_char_min=4_000,
                target_char_max=5_000, annotation_limit=2, run_id="pipeline",
            )
            self.assertEqual([call.args[2] for call in record.call_args_list], list(range(1, 8)))

    def test_multi_work_run_requires_explicit_skill_target(self) -> None:
        with patch("pipeline.pipeline.initialize_manifest"), \
             patch("pipeline.pipeline.active_work_ids", return_value=["work-001", "work-002"]):
            with self.assertRaisesRegex(ValueError, "只选择一部作品"):
                run_pipeline(PipelineOptions(author_id="Example", start_stage=6, end_stage=6))


if __name__ == "__main__":
    unittest.main()
