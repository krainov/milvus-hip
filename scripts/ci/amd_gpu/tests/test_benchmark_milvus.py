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

    def test_recall_postprocessing_is_outside_qps_timer(self) -> None:
        benchmark = (
            REPOSITORY_ROOT / "scripts/ci/amd_gpu/benchmark_milvus.py"
        ).read_text()
        timed_search = benchmark.split("def timed_search(", maxsplit=1)[
            1
        ].split("def result_row(", maxsplit=1)[0]

        self.assertIn(
            "elapsed = time.perf_counter() - started_all", timed_search
        )
        self.assertNotIn("search_ids", timed_search)

    def test_long_run_can_interleave_nprobes(self) -> None:
        benchmark = (
            REPOSITORY_ROOT / "scripts/ci/amd_gpu/benchmark_milvus.py"
        ).read_text()

        self.assertIn('"--interleave-nprobes"', benchmark)
        self.assertIn(
            "rotating-interleaved-nprobe-rounds-full-batch-client-call-v1",
            benchmark,
        )
        self.assertIn("offset = run_index % len(nprobes)", benchmark)
        self.assertIn(
            '"recall@10": median_measurement(recall_samples)', benchmark
        )

    def test_workflow_accepts_nightly_gist_run_count(self) -> None:
        workflow = (
            REPOSITORY_ROOT / ".github/workflows/amd-gpu-parity.yaml"
        ).read_text()

        self.assertIn("gist_runs:", workflow)
        self.assertIn("inputs.gist_runs || '10'", workflow)
        self.assertEqual(2, workflow.count('--runs "${GIST_BENCHMARK_RUNS}"'))
        self.assertEqual(
            2,
            workflow.count('--runs "${GIST_FLAT_BENCHMARK_RUNS}"'),
        )
        self.assertEqual(
            4,
            workflow.count(
                '--warmup-runs "${GIST_BENCHMARK_WARMUP_RUNS}"'
            ),
        )
        self.assertEqual(
            4,
            workflow.count('--query-batch "${GIST_BENCHMARK_QUERY_BATCH}"'),
        )
        self.assertEqual(2, workflow.count("--interleave-nprobes"))
        self.assertEqual(2, workflow.count("--recall-runs 10"))
        self.assertEqual(
            2,
            workflow.count("if (( GIST_BENCHMARK_RUNS >= 1000 )); then"),
        )
        self.assertIn("benchmark-amd-gist-ivf-flat.json", workflow)
        self.assertIn("benchmark-nvidia-gist-ivf-flat.json", workflow)
        self.assertIn('MILVUS_CI_GPU_INIT_MEM_SIZE: "0"', workflow)
        self.assertIn('MILVUS_CI_GPU_MAX_MEM_SIZE: "0"', workflow)

    def test_ci_milvus_can_use_available_gpu_memory_for_gist_flat(self) -> None:
        service = (
            REPOSITORY_ROOT / "scripts/ci/amd_gpu/milvus_service.sh"
        ).read_text()

        self.assertIn(
            'GPU_INITMEMSIZE=${MILVUS_CI_GPU_INIT_MEM_SIZE:-2048}', service
        )
        self.assertIn(
            'GPU_MAXMEMSIZE=${MILVUS_CI_GPU_MAX_MEM_SIZE:-4096}', service
        )

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
