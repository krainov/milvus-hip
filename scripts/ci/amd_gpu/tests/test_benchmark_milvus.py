from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.ci.amd_gpu.common import verify_sha256


class DatasetVerificationTest(unittest.TestCase):
    def test_accepts_expected_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.hdf5"
            dataset.write_bytes(b"frozen dataset")

            actual = verify_sha256(
                dataset,
                "88c08a34ebfe74a6a202127e9af097dea312b0c5c71eec87b79cba3854d17603",
                "dataset",
            )

        self.assertEqual(
            "88c08a34ebfe74a6a202127e9af097dea312b0c5c71eec87b79cba3854d17603",
            actual,
        )

    def test_rejects_unexpected_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.hdf5"
            dataset.write_bytes(b"changed dataset")

            with self.assertRaisesRegex(ValueError, "does not match expected"):
                verify_sha256(dataset, "0" * 64, "dataset")


if __name__ == "__main__":
    unittest.main()
