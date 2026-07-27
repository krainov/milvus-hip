from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.ci.amd_gpu.build_sut import (
    BuildError,
    package_libraries,
    verify_artifact,
    verify_checkout,
    verify_dependency_source,
)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class BuildInputTest(unittest.TestCase):
    def make_repo(self, root: Path) -> str:
        git(root, "init", "-q")
        git(root, "config", "user.name", "CI Test")
        git(root, "config", "user.email", "ci@example.invalid")
        (root / "tracked.txt").write_text("source\n")
        git(root, "add", "tracked.txt")
        git(root, "commit", "-qm", "source")
        return git(root, "rev-parse", "HEAD")

    def test_checkout_requires_requested_commit_and_clean_tracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commit = self.make_repo(root)
            with mock.patch.dict(os.environ, {"GITHUB_SHA": commit}):
                self.assertEqual((commit, commit), verify_checkout(root))

            (root / "tracked.txt").write_text("changed\n")
            with mock.patch.dict(os.environ, {"GITHUB_SHA": commit}):
                with self.assertRaisesRegex(BuildError, "tracked changes"):
                    verify_checkout(root)

    def test_dependency_requires_pinned_clean_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commit = self.make_repo(root)
            result = verify_dependency_source(root, commit)
            self.assertEqual(commit, result["commit"])

            with self.assertRaisesRegex(BuildError, "expected"):
                verify_dependency_source(root, "0" * 40)

    def test_artifact_is_verified_by_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "libcuvs.so"
            artifact.write_bytes(b"pinned")
            expected = hashlib.sha256(b"pinned").hexdigest()
            self.assertEqual(expected, verify_artifact(artifact, expected)["sha256"])

    def test_runtime_libraries_are_packaged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "internal/core/output/lib"
            source.mkdir(parents=True)
            (source / "libmilvus_core.so").write_bytes(b"library")

            result = package_libraries(root)

            self.assertIn("libmilvus_core.so", result["files"])
            self.assertEqual(
                b"library",
                (root / "lib/libmilvus_core.so").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
