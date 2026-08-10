from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPOSITORY_ROOT / "scripts/ci/amd_gpu/evidence.py"


class EvidencePackTest(unittest.TestCase):
    def test_non_gating_failure_is_retained_without_failing_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            required = root / "required.json"
            informational = root / "informational.json"
            required.write_text(json.dumps({"schema": "test.v1", "status": "pass"}))
            informational.write_text(
                json.dumps({"schema": "test.v1", "status": "fail"})
            )
            output = root / "pack"

            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--run-id",
                    "123",
                    "--input",
                    f"required={required}",
                    "--input",
                    f"informational={informational}",
                    "--non-gating",
                    "informational",
                    "--output-dir",
                    str(output),
                ],
                capture_output=True,
                check=False,
                text=True,
            )

            self.assertEqual(0, result.returncode, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual("pass", manifest["verdict"])
            self.assertEqual(["informational"], manifest["non_gating_evidence"])
            items = {item["name"]: item for item in manifest["evidence"]}
            self.assertTrue(items["required"]["gating"])
            self.assertFalse(items["informational"]["gating"])
            self.assertEqual("fail", items["informational"]["status"])

    def test_unknown_non_gating_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = root / "payload.json"
            payload.write_text(json.dumps({"schema": "test.v1", "status": "pass"}))
            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--run-id",
                    "123",
                    "--input",
                    f"required={payload}",
                    "--non-gating",
                    "missing",
                    "--output-dir",
                    str(root / "pack"),
                ],
                capture_output=True,
                check=False,
                text=True,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("not present in --input", result.stderr)


if __name__ == "__main__":
    unittest.main()
