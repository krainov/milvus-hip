from __future__ import annotations

import unittest

from scripts.ci.amd_gpu.triage import classify, redact


class TriageTest(unittest.TestCase):
    def test_redacts_known_credentials(self) -> None:
        text = "Authorization: Bearer abcdefghijklmnop password=secret-value"  # pragma: allowlist secret
        redacted = redact(text)
        self.assertNotIn("abcdefghijklmnop", redacted)
        self.assertNotIn("secret-value", redacted)

    def test_classifies_gpu_contention(self) -> None:
        matches = classify("GPU has active compute processes")
        self.assertEqual("gpu_busy", matches[0]["signature"])

    def test_green_doctor_tool_names_are_not_failures(self) -> None:
        self.assertEqual([], classify("[PASS] hipcc\n[PASS] nvcc\nQPS: 1000"))

    def test_successful_docker_pull_is_not_a_failure(self) -> None:
        self.assertEqual([], classify("+ docker pull image\nStatus: up to date"))

    def test_classifies_node_identity_failure(self) -> None:
        matches = classify(
            "rpc error: node not match[expectedNodeID=17][actualNodeID=18]"
        )
        self.assertEqual("milvus_node_mismatch", matches[0]["signature"])

    def test_classifies_dependency_revision_mismatch(self) -> None:
        matches = classify(
            "dependency /opt/knowhere is deadbee, expected 0123456789abcdef"
        )
        self.assertEqual("dependency_revision_mismatch", matches[0]["signature"])

    def test_classifies_missing_exact_sut(self) -> None:
        matches = classify("Exact-SHA Milvus image is missing: milvus-ci:test")
        self.assertEqual("exact_sut_missing", matches[0]["signature"])


if __name__ == "__main__":
    unittest.main()
