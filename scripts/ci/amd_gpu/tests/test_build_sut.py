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
    nvidia_compose_build_command,
    package_libraries,
    prepare_nvidia_cache,
    prepare_nvidia_cmake,
    prepare_nvidia_go_toolchain,
    prepare_nvidia_rust_toolchain,
    prepare_nvidia_xxhash,
    secure_cached_conan_sources,
    verify_artifact,
    verify_checkout,
    verify_ci_workspace,
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

    def test_ci_cleanup_is_confined_to_runner_workspace(self) -> None:
        root = Path("/runner/_work/milvus/milvus")
        with mock.patch.dict(
            os.environ,
            {
                "GITHUB_ACTIONS": "true",
                "GITHUB_WORKSPACE": str(root),
                "RUNNER_WORKSPACE": "/runner/_work/milvus",
            },
        ):
            result = verify_ci_workspace(root)

        self.assertTrue(result["enforced"])

    def test_ci_rejects_source_outside_runner_workspace(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "GITHUB_ACTIONS": "true",
                "GITHUB_WORKSPACE": "/home/developer/milvus",
                "RUNNER_WORKSPACE": "/runner/_work/milvus",
            },
        ):
            with self.assertRaisesRegex(BuildError, "outside runner workspace"):
                verify_ci_workspace(Path("/home/developer/milvus"))

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
                    "ubuntu22.04-cmake-3.30.4",
                    "ubuntu22.04-conan",
                    "ubuntu22.04-go-mod",
                    "ubuntu22.04-go-sumdb",
                    "ubuntu22.04-rust-1.73.0",
                    "ubuntu22.04-xxhash-0.8.2",
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
        self.assertIn(
            "CONAN_USER_HOME=/home/milvus",
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

    def test_nvidia_cmake_is_pinned_and_verified_in_builder(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch("scripts.ci.amd_gpu.build_sut.os.getuid", return_value=1001),
            mock.patch("scripts.ci.amd_gpu.build_sut.os.getgid", return_value=1002),
            mock.patch("scripts.ci.amd_gpu.build_sut.run_live") as run_live,
            mock.patch(
                "scripts.ci.amd_gpu.build_sut.run_command",
                return_value={
                    "returncode": 0,
                    "stdout": "cmake version 3.30.4\n",
                    "stderr": "",
                },
            ) as run_command,
        ):
            install_root = Path(temporary) / "cmake"
            result = prepare_nvidia_cmake(
                Path("/source"),
                "builder:tag",
                install_root,
            )

        install_command = run_live.call_args.args[0]
        verify_command = run_command.call_args.args[0]
        self.assertIn("cmake==3.30.4", install_command)
        self.assertIn(f"{install_root}:/tmp/.local", install_command)
        self.assertIn("/tmp/.local/bin/cmake", verify_command)
        self.assertEqual("cmake version 3.30.4", result["reported"])

    def test_nvidia_xxhash_is_built_static_from_pinned_source(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch("scripts.ci.amd_gpu.build_sut.os.getuid", return_value=1001),
            mock.patch("scripts.ci.amd_gpu.build_sut.os.getgid", return_value=1002),
            mock.patch(
                "scripts.ci.amd_gpu.build_sut.verify_dependency_source",
                return_value={"commit": "b" * 40, "clean": True},
            ),
            mock.patch("scripts.ci.amd_gpu.build_sut.run_live") as run_live,
        ):
            temporary_root = Path(temporary)
            install_root = temporary_root / "cache/xxhash"
            config = install_root / "lib/cmake/xxHash/xxHashConfig.cmake"
            library = install_root / "lib/libxxhash.a"
            config.parent.mkdir(parents=True)
            config.write_text("config\n")
            library.write_bytes(b"static")

            result = prepare_nvidia_xxhash(
                Path("/source"),
                "builder:tag",
                Path("/ci/xxhash"),
                install_root,
                "b" * 40,
            )

        configure_command = run_live.call_args_list[0].args[0]
        install_command = run_live.call_args_list[1].args[0]
        self.assertIn("/ci/xxhash:/src/xxhash:ro", configure_command)
        self.assertIn("-DBUILD_SHARED_LIBS=OFF", configure_command)
        self.assertIn("-DXXHASH_BUILD_XXHSUM=OFF", configure_command)
        self.assertIn("-DCMAKE_POSITION_INDEPENDENT_CODE=ON", configure_command)
        self.assertIn("install", install_command)
        self.assertEqual("b" * 40, result["source"]["commit"])

    def test_nvidia_rust_toolchain_is_rehomed_for_runner_user(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch("scripts.ci.amd_gpu.build_sut.os.getuid", return_value=1001),
            mock.patch("scripts.ci.amd_gpu.build_sut.os.getgid", return_value=1002),
            mock.patch("scripts.ci.amd_gpu.build_sut.run_live") as run_live,
            mock.patch(
                "scripts.ci.amd_gpu.build_sut.run_command",
                side_effect=[
                    {
                        "returncode": 0,
                        "stdout": "CUDA banner\ncargo 1.73.0 (test)",
                        "stderr": "",
                    },
                    {
                        "returncode": 0,
                        "stdout": "CUDA banner\nrustc 1.73.0 (test)",
                        "stderr": "",
                    },
                ],
            ),
        ):
            install_root = Path(temporary) / "rust"
            result = prepare_nvidia_rust_toolchain(
                Path("/source"),
                "builder:tag",
                install_root,
                "sha256:" + "a" * 64,
            )

        copy_command = run_live.call_args.args[0]
        self.assertIn(f"{install_root}:/opt/milvus-ci/rust", copy_command)
        self.assertIn("CI_UID=1001", copy_command)
        self.assertIn("CI_GID=1002", copy_command)
        self.assertEqual("cargo 1.73.0 (test)", result["cargo"])

    def test_nvidia_build_reuses_preseeded_sumdb(self) -> None:
        with (
            mock.patch("scripts.ci.amd_gpu.build_sut.os.getuid", return_value=1001),
            mock.patch("scripts.ci.amd_gpu.build_sut.os.getgid", return_value=1002),
        ):
            command = nvidia_compose_build_command(
                Path("/cache/sumdb"),
                Path("/cache/cmake"),
                Path("/cache/xxhash"),
                Path("/cache/rust"),
            )

        self.assertIn("1001:1002", command)
        self.assertIn("/cache/sumdb:/go/pkg/sumdb", command)
        self.assertIn("HOME=/tmp", command)
        self.assertIn("ENABLE_AZURE=false", command)
        self.assertIn("GOTOOLCHAIN=go1.22.0", command)
        self.assertIn("/cache/cmake:/tmp/.local:ro", command)
        self.assertIn("/cache/xxhash:/opt/milvus-ci/xxhash:ro", command)
        self.assertIn("/cache/rust:/opt/milvus-ci/rust", command)
        self.assertIn("CARGO_HOME=/opt/milvus-ci/rust/cargo", command)
        self.assertIn("RUSTUP_HOME=/opt/milvus-ci/rust/rustup", command)
        self.assertTrue(
            any(value.startswith("PATH=/tmp/.local/bin:") for value in command)
        )
        self.assertTrue(any("/usr/local/go/bin" in value for value in command))
        self.assertEqual(["gpubuilder", "make", "milvus-gpu"], command[-3:])

    def test_nvidia_build_mounts_pinned_knowhere_source(self) -> None:
        with (
            mock.patch("scripts.ci.amd_gpu.build_sut.os.getuid", return_value=1001),
            mock.patch("scripts.ci.amd_gpu.build_sut.os.getgid", return_value=1002),
        ):
            command = nvidia_compose_build_command(
                Path("/cache/sumdb"),
                Path("/cache/cmake"),
                Path("/cache/xxhash"),
                Path("/cache/rust"),
                Path("/ci/knowhere"),
            )

        self.assertIn("/ci/knowhere:/opt/milvus-ci/knowhere:ro", command)
        self.assertIn(
            (
                "CMAKE_EXTRA_ARGS="
                "-DMILVUS_KNOWHERE_SOURCE_DIR=/opt/milvus-ci/knowhere "
                "-DMILVUS_WITH_HIP=OFF -DWITH_HIP=OFF -DWITH_CUVS=ON "
                "-DxxHash_DIR=/opt/milvus-ci/xxhash/lib/cmake/xxHash "
                "-DxxHash_INCLUDE_DIRS=/opt/milvus-ci/xxhash/include"
            ),
            command,
        )
        self.assertIn("MILVUS_WITH_HIP=0", command)
        self.assertIn("KNOWHERE_WITH_HIP=0", command)
        self.assertNotIn("TOKEN", " ".join(command).upper())

    def test_milvus_cmake_requires_explicit_hip_opt_in(self) -> None:
        root = Path(__file__).resolve().parents[4]
        cmake = (
            root / "internal/core/thirdparty/knowhere/CMakeLists.txt"
        ).read_text()
        core_cmake = (root / "internal/core/src/CMakeLists.txt").read_text()

        self.assertNotIn('set(WITH_HIP ON CACHE BOOL "" FORCE', cmake)
        self.assertIn("DEFINED MILVUS_WITH_HIP", cmake)
        self.assertIn("MILVUS GPU: WITH_CUVS=ON WITH_HIP=OFF (CUDA)", cmake)
        self.assertIn(
            'if(MILVUS_GPU_VERSION STREQUAL "ON" AND WITH_HIP)',
            core_cmake,
        )
        self.assertIn("project(core CXX C CUDA)", core_cmake)

    def test_insecure_cached_conan_source_is_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = root / "data/lzo/2.10/_/_/export/conandata.yml"
            recipe.parent.mkdir(parents=True)
            recipe.write_text(
                "url: http://www.oberhumer.com/opensource/lzo/download/"
                "lzo-2.10.tar.gz\n"
            )

            result = secure_cached_conan_sources(root)

            self.assertEqual("updated-to-https", result[0]["status"])
            self.assertIn("https://www.oberhumer.com", recipe.read_text())

    def test_missing_conan_recipe_is_deferred_until_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = secure_cached_conan_sources(Path(temporary))

        self.assertTrue(
            all(item["status"] == "recipe-not-cached" for item in result)
        )


if __name__ == "__main__":
    unittest.main()
