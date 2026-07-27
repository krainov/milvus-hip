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
    cleanup_stale_nvidia_builders,
    package_libraries,
    prepare_nvidia_cache,
    prepare_nvidia_go_toolchain,
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

    def test_nvidia_cache_matches_compose_volume_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            result = prepare_nvidia_cache(root, "ubuntu22.04")

            self.assertEqual(
                {
                    "ubuntu22.04-ccache",
                    "ubuntu22.04-conan",
                    "ubuntu22.04-go-mod",
                    "ubuntu22.04-go-sumdb",
                    "ubuntu22.04-vscode-extensions",
                },
                {Path(path).name for path in result["directories"]},
            )
            for path in result["directories"]:
                self.assertEqual(0o777, Path(path).stat().st_mode & 0o777)

    def test_stale_nvidia_builders_and_locks_are_removed(self) -> None:
        with (
            mock.patch(
                "scripts.ci.amd_gpu.build_sut.run_command",
                return_value={
                    "returncode": 0,
                    "stdout": "container-one\ncontainer-two",
                    "stderr": "",
                },
            ),
            mock.patch("scripts.ci.amd_gpu.build_sut.run_live") as run_live,
        ):
            result = cleanup_stale_nvidia_builders(
                Path("/source"),
                "builder:tag",
                Path("/cache/conan"),
            )

        self.assertEqual(
            ["container-one", "container-two"],
            result["removed_builder_containers"],
        )
        self.assertEqual(
            ["docker", "rm", "--force", "container-one", "container-two"],
            run_live.call_args_list[0].args[0],
        )
        self.assertIn(
            "/cache/conan:/home/milvus/.conan",
            run_live.call_args_list[1].args[0],
        )

    def test_go_toolchain_uses_runner_identity_and_writable_caches(self) -> None:
        with (
            mock.patch("scripts.ci.amd_gpu.build_sut.os.getuid", return_value=1001),
            mock.patch("scripts.ci.amd_gpu.build_sut.os.getgid", return_value=1002),
            mock.patch("scripts.ci.amd_gpu.build_sut.run_live") as run_live,
        ):
            result = prepare_nvidia_go_toolchain(
                Path("/source"),
                "builder:tag",
                Path("/cache/mod"),
                Path("/cache/sumdb"),
            )

        command = run_live.call_args.args[0]
        self.assertIn("1001:1002", command)
        self.assertIn("/cache/mod:/go/pkg/mod", command)
        self.assertIn("/cache/sumdb:/go/pkg/sumdb", command)
        self.assertEqual("go1.22.0", result["version"])


if __name__ == "__main__":
    unittest.main()
