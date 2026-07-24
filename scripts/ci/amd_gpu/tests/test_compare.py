from __future__ import annotations

import unittest

from scripts.ci.amd_gpu.compare import compare_benchmarks, compare_gtests


class BenchmarkComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        common = {
            "index_type": "GPU_IVF_PQ",
            "nlist": 1024,
            "m": 32,
            "nbits": 8,
            "k": 10,
            "dataset_sha256": "abc",
        }
        self.amd = {
            **common,
            "nprobe_results": [
                {"nprobe": 1, "qps": 200.0, "recall@10": 0.40},
                {"nprobe": 4, "qps": 180.0, "recall@10": 0.70},
            ],
        }
        self.nvidia = {
            **common,
            "nprobe_results": [
                {"nprobe": 1, "qps": 100.0, "recall@10": 0.401},
                {"nprobe": 4, "qps": 200.0, "recall@10": 0.699},
            ],
        }

    def test_matching_recall_passes(self) -> None:
        result = compare_benchmarks(self.amd, self.nvidia, 0.005)
        self.assertEqual("pass", result["status"])
        self.assertAlmostEqual(2.0, result["rows"][0]["qps_ratio_amd_to_nvidia"])

    def test_recall_regression_fails(self) -> None:
        self.amd["nprobe_results"][1]["recall@10"] = 0.60
        result = compare_benchmarks(self.amd, self.nvidia, 0.005)
        self.assertEqual("fail", result["status"])
        self.assertEqual([4], result["correctness_gate"]["failed_nprobes"])

    def test_failed_source_provenance_fails(self) -> None:
        self.amd["status"] = "fail"
        result = compare_benchmarks(self.amd, self.nvidia, 0.005)
        self.assertEqual("fail", result["status"])
        self.assertEqual(["amd"], result["input_failures"])


class GtestComparisonTest(unittest.TestCase):
    def test_shared_passing_case(self) -> None:
        amd = {
            "suite": [
                {
                    "binary": "TEST",
                    "cases": [{"name": "Suite.Case", "status": "PASS", "ms": 20}],
                }
            ]
        }
        nvidia = {
            "suite": [
                {
                    "binary": "TEST",
                    "cases": [{"name": "Suite.Case", "status": "PASS", "ms": 10}],
                }
            ]
        }
        result = compare_gtests(amd, nvidia)
        self.assertEqual("pass", result["status"])
        self.assertEqual(2.0, result["summary"]["timing_ratio_geomean_amd_to_nvidia"])

    def test_backend_failure_is_visible_in_case_comparison(self) -> None:
        amd = {
            "suite": [
                {
                    "binary": "CAGRA_C_TEST",
                    "cases": [
                        {"name": "CagraC.BuildSearch", "status": "FAIL", "ms": 20}
                    ],
                }
            ]
        }
        nvidia = {
            "suite": [
                {
                    "binary": "CAGRA_C_TEST",
                    "cases": [
                        {"name": "CagraC.BuildSearch", "status": "PASS", "ms": 10}
                    ],
                }
            ]
        }
        result = compare_gtests(amd, nvidia)
        self.assertEqual("fail", result["status"])
        self.assertEqual(
            ["CAGRA_C_TEST::CagraC.BuildSearch"],
            result["failed_cases"],
        )


if __name__ == "__main__":
    unittest.main()
