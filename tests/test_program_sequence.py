from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.jsonio import read_json, write_json
from pipeline.program_sequence import generate_program_sequence


class ProgramSequenceTests(unittest.TestCase):
    def test_only_a_verified_exit_state_is_passed_to_the_next_chapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            programs = []
            for number in range(1, 4):
                path = base / f"chapter-{number:04d}.program.json"
                path.write_text("{}", encoding="utf-8")
                programs.append(path)
            graph_path = base / "narrative_graph.json"
            graph_path.write_text("{}", encoding="utf-8")
            index = base / "index.json"
            write_json(index, {
                "author_id": "Example", "generation_mode": "controlled_expansion", "passed": True,
                "narrative_graph": str(graph_path),
                "chapters": [
                    {"chapter_no": number, "chapter_id": f"chapter-{number}", "program": str(path)}
                    for number, path in enumerate(programs, start=1)
                ],
            })
            output_one = base / "generated-1"; output_one.mkdir()
            exit_one = output_one / "chapter_exit_state.json"
            write_json(exit_one, {
                "overall_passed": True,
                "graph_patch_committed": True,
                "state": {"已完成场景": ["scene-1"]},
            })
            output_two = base / "generated-2"; output_two.mkdir()
            exit_two = output_two / "chapter_exit_state.json"
            write_json(exit_two, {"overall_passed": False, "state": {"错误": True}})
            calls: list[Path | None] = []

            def fake_generate(_author: str, _program: Path, *, story_state_path: Path | None = None, **_kwargs: object) -> dict[str, object]:
                calls.append(story_state_path)
                if len(calls) == 2:
                    self.assertIsNotNone(story_state_path)
                    self.assertEqual(read_json(story_state_path), {"已完成场景": ["scene-1"]})
                    return {"output_dir": str(output_two), "chapter_exit_state": str(exit_two), "overall_passed": False}
                return {"output_dir": str(output_one), "chapter_exit_state": str(exit_one), "overall_passed": True}

            with patch("pipeline.program_sequence.runs_dir", return_value=base / "runs"), \
                 patch("pipeline.program_sequence.generate_from_program", side_effect=fake_generate):
                report = generate_program_sequence("Example", index, run_id="sequence")

            summary = read_json(report)
            self.assertEqual(len(calls), 2)
            self.assertEqual([item["status"] for item in summary["chapters"]], ["passed", "rejected", "not_started"])
            self.assertTrue(summary["blocked"])


if __name__ == "__main__":
    unittest.main()
