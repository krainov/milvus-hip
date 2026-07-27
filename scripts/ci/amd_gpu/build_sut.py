#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from .common import host_metadata, run_command, sha256, utc_now, write_json
except ImportError:
    from common import host_metadata, run_command, sha256, utc_now, write_json


class BuildError(RuntimeError):
    pass


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise BuildError(f"{name} is required")
    return value


def git_value(root: Path, *args: str) -> str:
    result = run_command(["git", "-C", str(root), *args])
    if result["returncode"] != 0:
        raise BuildError(result["stderr"] or f"git {' '.join(args)} failed")
    return result["stdout"]


def verify_checkout(root: Path) -> tuple[str, str]:
    source_commit = git_value(root, "rev-parse", "HEAD")
    requested_commit = os.environ.get("GITHUB_SHA", source_commit)
    if source_commit != requested_commit:
        raise BuildError(
            f"checkout commit {source_commit} does not match requested {requested_commit}"
        )
    tracked_changes = git_value(
        root, "status", "--porcelain", "--untracked-files=no"
    )
    if tracked_changes:
        raise BuildError(f"checkout has tracked changes before build:\n{tracked_changes}")
    return source_commit, requested_commit


def verify_dependency_source(path: Path, expected_commit: str) -> dict[str, Any]:
    actual_commit = git_value(path, "rev-parse", "HEAD")
    tracked_changes = git_value(
        path, "status", "--porcelain", "--untracked-files=no"
    )
    if actual_commit != expected_commit:
        raise BuildError(
            f"dependency {path} is {actual_commit}, expected {expected_commit}"
        )
    if tracked_changes:
        raise BuildError(f"dependency {path} has tracked changes:\n{tracked_changes}")
    return {
        "path": str(path),
        "commit": actual_commit,
        "clean": True,
    }


def verify_artifact(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise BuildError(f"required artifact is missing: {path}")
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise BuildError(
            f"artifact {path} has SHA256 {actual_sha256}, expected {expected_sha256}"
        )
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "size_bytes": path.stat().st_size,
    }


def run_live(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    print(f"+ {shlex.join(argv)}", flush=True)
    subprocess.run(argv, cwd=cwd, env=env, check=True)


def clean_outputs(root: Path) -> None:
    for relative in ("cmake_build", "internal/core/output", "bin", "lib"):
        path = root / relative
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def package_libraries(root: Path) -> dict[str, Any]:
    destination = root / "lib"
    destination.mkdir()
    copied: list[str] = []
    for relative in ("internal/core/output/lib", "internal/core/output/lib64"):
        source = root / relative
        if not source.is_dir():
            continue
        for path in source.iterdir():
            target = destination / path.name
            if path.is_dir():
                shutil.copytree(path, target, symlinks=True, dirs_exist_ok=True)
            elif path.is_symlink():
                target.symlink_to(os.readlink(path))
            elif path.is_file():
                shutil.copy2(path, target)
            copied.append(path.name)
    if not copied:
        raise BuildError("Milvus build produced no runtime libraries")
    return {
        "path": str(destination),
        "files": sorted(set(copied)),
    }


def find_xxhash_dir(prefix: Path) -> Path:
    candidates = [
        prefix / "lib/cmake/xxHash",
        prefix / "lib64/cmake/xxHash",
    ]
    for candidate in candidates:
        if (candidate / "xxHashConfig.cmake").is_file():
            return candidate
    raise BuildError(f"xxHashConfig.cmake is missing under {prefix}")


def build_amd(root: Path) -> dict[str, Any]:
    hip_prefix = Path(required_env("GPU_HIP_PREFIX")).resolve()
    knowhere_source = Path(required_env("GPU_KNOWHERE_SOURCE_DIR")).resolve()
    knowhere = verify_dependency_source(
        knowhere_source,
        required_env("GPU_KNOWHERE_COMMIT"),
    )
    hip_artifact = verify_artifact(
        hip_prefix / "lib/libcuvs.so",
        required_env("GPU_LIBCUVS_SHA256"),
    )
    xxhash_dir = find_xxhash_dir(hip_prefix)

    build_env = os.environ.copy()
    path_parts = [
        str(Path.home() / ".local/bin"),
        "/usr/local/go/bin",
        "/opt/rocm/bin",
        "/opt/rocm/llvm/bin",
        build_env.get("PATH", ""),
    ]
    library_parts = [
        str(hip_prefix / "lib"),
        "/opt/rocm/lib",
        build_env.get("LD_LIBRARY_PATH", ""),
    ]
    build_env.update(
        {
            "PATH": ":".join(path_parts),
            "LD_LIBRARY_PATH": ":".join(library_parts),
            "MILVUS_HIP_INSTALL_PREFIX": str(hip_prefix),
            "ROCMDS_INSTALL_PREFIX": str(hip_prefix),
            "ROCM_PATH": os.environ.get("ROCM_PATH", "/opt/rocm"),
            "HIP_PATH": os.environ.get("HIP_PATH", "/opt/rocm"),
            "ENABLE_AZURE": "false",
            "CC": os.environ.get("CC", "/usr/bin/cc"),
            "CXX": os.environ.get("CXX", "/opt/rocm/llvm/bin/clang++"),
            "CMAKE_PREFIX_PATH": (
                f"{hip_prefix};/opt/rocm;"
                f"{build_env.get('CMAKE_PREFIX_PATH', '')}"
            ),
            "CMAKE_EXTRA_ARGS": (
                f"-DMILVUS_KNOWHERE_SOURCE_DIR={knowhere_source} "
                f"-DxxHash_DIR={xxhash_dir}"
            ),
        }
    )
    run_live(["make", "milvus-gpu"], cwd=root, env=build_env)

    return {
        "builder": "host-rocm",
        "knowhere": knowhere,
        "hip_runtime": hip_artifact,
        "hip_prefix": str(hip_prefix),
    }


def inspect_image(reference: str) -> dict[str, Any]:
    result = run_command(
        [
            "docker",
            "image",
            "inspect",
            reference,
            "--format",
            "{{.Id}} {{json .RepoDigests}}",
        ]
    )
    if result["returncode"] != 0:
        raise BuildError(result["stderr"] or f"cannot inspect image {reference}")
    image_id, _, raw_digests = result["stdout"].partition(" ")
    return {
        "reference": reference,
        "id": image_id,
        "repo_digests": json.loads(raw_digests),
    }


def build_nvidia(root: Path, source_commit: str) -> dict[str, Any]:
    builder_image = required_env("NVIDIA_BUILDER_IMAGE")
    builder_digest = required_env("NVIDIA_BUILDER_DIGEST")
    runtime_base = required_env("NVIDIA_RUNTIME_BASE_IMAGE")
    runtime_image = required_env("MILVUS_CI_IMAGE")
    cache_root = Path(required_env("GPU_BUILD_CACHE_ROOT")).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    run_live(["docker", "pull", builder_image], cwd=root)
    builder = inspect_image(builder_image)
    if not any(item.endswith(f"@{builder_digest}") for item in builder["repo_digests"]):
        raise BuildError(
            f"builder image digest does not contain pinned {builder_digest}: "
            f"{builder['repo_digests']}"
        )

    build_env = os.environ.copy()
    build_env.update(
        {
            "DOCKER_VOLUME_DIRECTORY": str(cache_root),
            "IMAGE_REPO": "milvusdb",
            "OS_NAME": "ubuntu22.04",
            "GPU_DATE_VERSION": "20240806-d8668fe",
            "LATEST_GPU_DATE_VERSION": "20240806-d8668fe",
            "IS_NETWORK_MODE_HOST": "false",
            "ENABLE_AZURE": "false",
        }
    )
    run_live(
        ["bash", "build/builder_gpu.sh", "make", "milvus-gpu"],
        cwd=root,
        env=build_env,
    )
    runtime_libraries = package_libraries(root)
    builder_after_build = inspect_image(builder_image)
    if builder_after_build["id"] != builder["id"]:
        raise BuildError("NVIDIA builder image changed during the build")

    subprocess.run(
        ["docker", "image", "rm", "--force", runtime_image],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    run_live(
        [
            "docker",
            "build",
            "--network",
            "host",
            "--file",
            "scripts/ci/amd_gpu/Dockerfile.nvidia-runtime",
            "--build-arg",
            f"BASE_IMAGE={runtime_base}",
            "--build-arg",
            f"SOURCE_COMMIT={source_commit}",
            "--tag",
            runtime_image,
            ".",
        ],
        cwd=root,
    )
    return {
        "builder": builder,
        "runtime_base_image": runtime_base,
        "runtime_image": inspect_image(runtime_image),
        "runtime_libraries": runtime_libraries,
    }


def binary_metadata(root: Path) -> dict[str, Any]:
    binary = root / "bin/milvus"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise BuildError(f"Milvus binary was not produced: {binary}")
    return {
        "path": str(binary),
        "sha256": sha256(binary),
        "size_bytes": binary.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an exact-SHA GPU Milvus SUT")
    parser.add_argument("--backend", choices=("amd", "nvidia"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.source_root.resolve()
    started = time.monotonic()
    payload: dict[str, Any] = {
        "schema": "milvus.gpu-build.v1",
        "generated_at": utc_now(),
        "backend": args.backend,
        "host": host_metadata(),
        "status": "fail",
    }
    try:
        source_commit, requested_commit = verify_checkout(root)
        payload.update(
            {
                "source_commit": source_commit,
                "requested_commit": requested_commit,
                "source_clean_before_build": True,
            }
        )
        clean_outputs(root)
        if args.backend == "amd":
            build_details = build_amd(root)
        else:
            build_details = build_nvidia(root, source_commit)
        payload.update(
            {
                "status": "pass",
                "binary": binary_metadata(root),
                "build": build_details,
            }
        )
    except (BuildError, OSError, subprocess.CalledProcessError) as error:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "returncode": getattr(error, "returncode", None),
        }
    finally:
        payload["duration_s"] = time.monotonic() - started
        write_json(args.output, payload)

    print(f"{args.backend} exact-SHA build: {payload['status']}")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
