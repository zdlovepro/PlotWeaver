from __future__ import annotations

import unittest
from unittest.mock import patch

from pipeline.program_fidelity import _SCORE_KEYS, _judge_structure, hard_structure_pass


class ProgramFidelityTests(unittest.TestCase):
    def _judgment(self) -> dict[str, object]:
        return {
            "评分": {key: 4 for key in _SCORE_KEYS},
            "遗漏": [], "矛盾": [], "优点": ["结构完整"], "下一步修复": [],
        }

    def test_hard_pass_requires_all_dimensions_and_no_explicit_gap(self) -> None:
        self.assertTrue(hard_structure_pass(self._judgment()))
        missing = self._judgment(); missing["遗漏"] = ["遗漏结果"]
        self.assertFalse(hard_structure_pass(missing))
        low_score = self._judgment(); low_score["评分"] = {**low_score["评分"], "逻辑自洽": 3}
        self.assertFalse(hard_structure_pass(low_score))

    def test_model_judgment_uses_a_strict_chinese_json_contract(self) -> None:
        with patch("pipeline.program_fidelity.complete_json", return_value=self._judgment()):
            judgment = _judge_structure({"事件": []}, {"场景程序": []}, "生成正文", {"overall_passed": True})
        self.assertEqual(judgment, self._judgment())
        invalid = self._judgment(); invalid["额外字段"] = True
        with patch("pipeline.program_fidelity.complete_json", return_value=invalid):
            with self.assertRaises(ValueError):
                _judge_structure({"事件": []}, {"场景程序": []}, "生成正文", {})


if __name__ == "__main__":
    unittest.main()
