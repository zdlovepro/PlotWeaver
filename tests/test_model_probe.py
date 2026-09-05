from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.jsonio import read_json
from pipeline.model import ModelSettings
from pipeline.model_probe import probe_model


class ModelProbeTests(unittest.TestCase):
    def test_probe_persists_a_passed_nonempty_json_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            with patch("pipeline.model_probe.runs_dir", return_value=base), \
                 patch("pipeline.model_probe.ModelSettings.from_environment", return_value=ModelSettings(api_key="test")), \
                 patch("pipeline.model_probe.complete_json", return_value={"状态": "可用", "协议": "json_object"}):
                target = probe_model("Example", run_id="probe")
            self.assertEqual(read_json(target)["status"], "passed")

    def test_probe_persists_failure_before_reraising(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            with patch("pipeline.model_probe.runs_dir", return_value=base), \
                 patch("pipeline.model_probe.ModelSettings.from_environment", return_value=ModelSettings(api_key="test")), \
                 patch("pipeline.model_probe.complete_json", side_effect=RuntimeError("模型返回空内容")):
                with self.assertRaisesRegex(RuntimeError, "空内容"):
                    probe_model("Example", run_id="probe")
            self.assertEqual(read_json(base / "model_probe.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
