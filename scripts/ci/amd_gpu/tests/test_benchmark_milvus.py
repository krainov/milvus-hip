from __future__ import annotations

import tempfile
import unittest
from argparse import ArgumentTypeError
from pathlib import Path

from scripts.ci.amd_gpu.common import (
    median_measurement,
    positive_int,
    verify_sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


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


class BenchmarkRunsTest(unittest.TestCase):
    def test_accepts_positive_run_count(self) -> None:
        self.assertEqual(1000, positive_int("1000"))

    def test_rejects_non_positive_run_count(self) -> None:
        with self.assertRaises(ArgumentTypeError):
            positive_int("0")

    def test_uses_median_for_run_measurements(self) -> None:
        self.assertEqual(25.0, median_measurement([10.0, 20.0, 30.0, 1000.0]))

    def test_workflow_accepts_nightly_gist_run_count(self) -> None:
        workflow = (
            REPOSITORY_ROOT / ".github/workflows/amd-gpu-parity.yaml"
        ).read_text()

        self.assertIn("gist_runs:", workflow)
        self.assertIn("inputs.gist_runs || '1'", workflow)
        self.assertEqual(2, workflow.count('--runs "${GIST_BENCHMARK_RUNS}"'))

    def test_amd_gtests_use_a_pinned_ci_owned_checkout(self) -> None:
        workflow = (
            REPOSITORY_ROOT / ".github/workflows/amd-gpu-parity.yaml"
        ).read_text()
        self.assertIn(
            "GPU_GTEST_DIR: /home/amd/milvus-ci/hipVS/cpp/build/gtests",
            workflow,
        )
        self.assertIn(
            "GPU_LIBRARY_SOURCE_DIR: /home/amd/milvus-ci/hipVS", workflow
        )
        self.assertIn("GPU_LIBRARY_COMMIT:", workflow)
        self.assertIn(
            'git -C "${GPU_LIBRARY_SOURCE_DIR}" diff --quiet HEAD --', workflow
        )


if __name__ == "__main__":
    unittest.main()
