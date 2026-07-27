#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    from .common import host_metadata, run_command, sha256, utc_now, write_json
except ImportError:
    from common import host_metadata, run_command, sha256, utc_now, write_json


DEFAULT_BINARIES = [
    "BRUTEFORCE_C_TEST",
    "IVF_FLAT_C_TEST",
    "IVF_PQ_C_TEST",
    "CAGRA_C_TEST",
    "DISTANCE_C_TEST",
]


def collect_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for suite in payload.get("testsuites", []):
        for test in suite.get("testsuite", []):
            failures = test.get("failures") or []
            skipped = test.get("status") == "NOTRUN" or test.get("result") == "SKIPPED"
            status = "SKIP" if skipped else ("FAIL" if failures else "PASS")
            elapsed = str(test.get("time", "0s")).removesuffix("s")
            try:
                milliseconds = float(elapsed) * 1000
            except ValueError:
                milliseconds = 0.0
            cases.append(
                {
                    "name": f"{suite.get('name', '')}.{test.get('name', '')}".strip("."),
                    "status": status,
                    "ms": milliseconds,
                    "failures": failures,
                }
            )
    return cases


def container_command(
    binary: Path,
    output_path: Path,
    gtest_filter: str | None,
    *,
    image: str,
    host_source: Path,
    container_source: Path,
    output_dir: Path,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "--volume",
        f"{host_source}:{container_source}:ro",
        "--volume",
        f"{output_dir}:/gtest-output",
        "--env",
        (
            "LD_LIBRARY_PATH="
            f"{container_source}/.ci-conda/lib:/opt/conda/lib:"
            "/opt/conda/targets/x86_64-linux/lib"
        ),
        "--entrypoint",
        str(binary),
        image,
        "--gtest_color=no",
        f"--gtest_output=json:/gtest-output/{output_path.name}",
    ]
    if gtest_filter:
        command.append(f"--gtest_filter={gtest_filter}")
    return command


def run_binary(
    binary: Path,
    gtest_filter: str | None,
    *,
    container_image: str | None,
    host_source: Path | None,
    container_source: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="milvus-gtest-") as temp_dir:
        output_path = Path(temp_dir) / "gtest.json"
        if container_image:
            if not host_source:
                raise ValueError("container gtests require --container-host-source")
            os.chmod(temp_dir, 0o777)
            command = container_command(
                binary,
                output_path,
                gtest_filter,
                image=container_image,
                host_source=host_source,
                container_source=container_source,
                output_dir=Path(temp_dir),
            )
        else:
            command = [
                str(binary),
                "--gtest_color=no",
                f"--gtest_output=json:{output_path}",
            ]
            if gtest_filter:
                command.append(f"--gtest_filter={gtest_filter}")

        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            returncode = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired as error:
            returncode = 124
            stdout = error.stdout or ""
            stderr = error.stderr or "gtest timed out after 3600s"
        payload = json.loads(output_path.read_text()) if output_path.exists() else {}
        cases = collect_cases(payload)
        return {
            "binary": binary.name,
            "filter": gtest_filter,
            "returncode": returncode,
            "stdout_tail": stdout[-20_000:],
            "stderr_tail": stderr[-20_000:],
            "cases": cases,
        }


def source_commit(source: Path | None) -> str | None:
    if not source:
        return None
    result = run_command(["git", "-C", str(source), "rev-parse", "HEAD"])
    return result["stdout"] if result["returncode"] == 0 else None


def source_dirty(source: Path | None) -> bool | None:
    if not source:
        return None
    result = run_command(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"]
    )
    return bool(result["stdout"]) if result["returncode"] == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a common cuVS/hipVS gtest set")
    parser.add_argument(
        "--gtest-dir",
        type=Path,
        default=os.environ.get("GPU_GTEST_DIR"),
        required=os.environ.get("GPU_GTEST_DIR") is None,
    )
    parser.add_argument("--backend", choices=("amd", "nvidia"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", action="append", dest="binaries")
    parser.add_argument("--gtest-filter")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=os.environ.get("GPU_LIBRARY_SOURCE_DIR"),
    )
    parser.add_argument(
        "--container-image",
        default=os.environ.get("GPU_GTEST_CONTAINER_IMAGE"),
    )
    parser.add_argument(
        "--container-host-source",
        type=Path,
        default=os.environ.get("GPU_GTEST_HOST_SOURCE_DIR"),
    )
    parser.add_argument(
        "--container-source",
        type=Path,
        default=Path(os.environ.get("GPU_GTEST_CONTAINER_SOURCE_DIR", "/workspace/cuvs")),
    )
    args = parser.parse_args()

    binaries = args.binaries or DEFAULT_BINARIES
    suites: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in binaries:
        binary = args.gtest_dir / name
        if not args.container_image and not binary.is_file():
            missing.append(name)
            continue
        suite = run_binary(
            binary,
            args.gtest_filter,
            container_image=args.container_image,
            host_source=args.container_host_source,
            container_source=args.container_source,
        )
        host_binary = (
            args.container_host_source / binary.relative_to(args.container_source)
            if args.container_image and args.container_host_source
            else binary
        )
        if host_binary.is_file():
            suite["binary_sha256"] = sha256(host_binary)
        suites.append(suite)

    all_cases = [case for suite in suites for case in suite["cases"]]
    failed = [case["name"] for case in all_cases if case["status"] == "FAIL"]
    failed_binaries = [
        suite["binary"] for suite in suites if suite["returncode"] != 0
    ]
    missing_is_failure = bool(missing and not args.allow_missing)
    status = (
        "pass"
        if all_cases and not failed and not failed_binaries and not missing_is_failure
        else "fail"
    )
    payload = {
        "schema": "milvus.gpu-gtest-run.v1",
        "generated_at": utc_now(),
        "backend": args.backend,
        "host": host_metadata(),
        "gtest_dir": str(args.gtest_dir),
        "library_source_commit": source_commit(args.source_dir),
        "library_source_dirty": source_dirty(args.source_dir),
        "container_image": args.container_image,
        "status": status,
        "n_cases": len(all_cases),
        "n_pass": sum(case["status"] == "PASS" for case in all_cases),
        "n_fail": len(failed),
        "missing_binaries": missing,
        "failed_binaries": failed_binaries,
        "suite": suites,
    }
    write_json(args.output, payload)
    print(
        f"{args.backend} gtests: {payload['n_pass']}/{payload['n_cases']} pass; "
        f"missing={len(missing)}"
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
