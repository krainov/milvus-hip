from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.ci.amd_gpu.secret_scan import scan_file


class SecretScanTest(unittest.TestCase):
    def test_ignores_environment_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "config.py"
            path.write_text('api_key = os.environ.get("API_KEY")\n')
            self.assertEqual([], scan_file(path, root))

    def test_finds_literal_credential(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "config.py"
            path.write_text('password = "literal-secret-value"\n')  # pragma: allowlist secret
            self.assertEqual(
                ["config.py:1: credential_assignment"],
                scan_file(path, root),
            )


if __name__ == "__main__":
    unittest.main()
