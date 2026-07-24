#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from .common import host_metadata, run_command, utc_now, write_json
except ImportError:
    from common import host_metadata, run_command, utc_now, write_json


def probe(
    name: str,
    argv: list[str],
    *,
    required: bool = True,
    timeout: int = 30,
) -> dict[str, Any]:
    result = run_command(argv, timeout=timeout)
    return {
        "name": name,
        "required": required,
        "passed": result["returncode"] == 0,
        **result,
    }


def parse_csv_line(text: str) -> list[str]:
    rows = list(csv.reader(io.StringIO(text)))
    return [value.strip() for value in rows[0]] if rows else []


def nvidia_checks(expected_gpu: str, require_idle: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks = [
        probe("docker", ["docker", "info", "--format", "{{.ServerVersion}}"]),
        probe(
            "nvidia-smi",
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,memory.used,compute_cap",
                "--format=csv,noheader,nounits",
            ],
        ),
        probe(
            "nvidia-container-runtime",
            ["nvidia-container-cli", "--version"],
        ),
    ]

    gpu: dict[str, Any] = {}
    smi = next(check for check in checks if check["name"] == "nvidia-smi")
    if smi["passed"]:
        fields = parse_csv_line(smi["stdout"].splitlines()[0])
        if len(fields) == 6:
            gpu = dict(
                zip(
                    [
                        "name",
                        "uuid",
                        "driver_version",
                        "memory_total_mib",
                        "memory_used_mib",
                        "compute_capability",
                    ],
                    fields,
                )
            )
        expected_passed = expected_gpu.lower() in gpu.get("name", "").lower()
        checks.append(
            {
                "name": "expected-gpu",
                "required": True,
                "passed": expected_passed,
                "stdout": gpu.get("name", ""),
                "stderr": "" if expected_passed else f"expected {expected_gpu}",
                "command": [],
                "returncode": 0 if expected_passed else 1,
            }
        )

    processes = probe(
        "gpu-compute-processes",
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        required=require_idle,
    )
    if processes["returncode"] == 0:
        processes["passed"] = not require_idle or not processes["stdout"].strip()
        if require_idle and not processes["passed"]:
            processes["stderr"] = "GPU has active compute processes"
    checks.append(processes)
    return checks, gpu


def amd_checks(expected_arch: str, require_idle: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks = [
        probe("docker", ["docker", "info", "--format", "{{.ServerVersion}}"]),
        probe("kfd-device", ["test", "-r", "/dev/kfd"]),
        probe("rocminfo", ["rocminfo"], timeout=60),
        probe("hipcc", ["hipcc", "--version"]),
        probe(
            "rocm-smi",
            ["rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "vram"],
        ),
    ]

    rocminfo = next(check for check in checks if check["name"] == "rocminfo")
    expected_passed = expected_arch in rocminfo["stdout"] if rocminfo["passed"] else False
    checks.append(
        {
            "name": "expected-gpu-arch",
            "required": True,
            "passed": expected_passed,
            "stdout": expected_arch if expected_passed else "",
            "stderr": "" if expected_passed else f"{expected_arch} was not found",
            "command": [],
            "returncode": 0 if expected_passed else 1,
        }
    )

    process_check = probe(
        "gpu-compute-processes",
        ["rocm-smi", "--showpids", "--json"],
        required=require_idle,
    )
    if process_check["returncode"] == 0 and require_idle:
        output = process_check["stdout"].strip()
        if not output or "No JSON data to report" in output:
            processes = {}
        else:
            try:
                processes = json.loads(output).get("system", {})
            except json.JSONDecodeError:
                processes = {"unparseable": output}
        has_processes = bool(processes)
        process_check["passed"] = not has_processes
        if has_processes:
            process_check["stderr"] = "GPU has active compute processes"
    checks.append(process_check)

    gpu = {
        "arch": expected_arch,
        "rocm_path": str(Path(shutil.which("hipcc") or "").resolve().parents[1])
        if shutil.which("hipcc")
        else None,
    }
    return checks, gpu


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a GPU CI runner")
    parser.add_argument("--backend", choices=("amd", "nvidia"), required=True)
    parser.add_argument("--expected", required=True, help="Expected GPU name or architecture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-idle", action="store_true")
    args = parser.parse_args()

    if args.backend == "nvidia":
        checks, gpu = nvidia_checks(args.expected, args.require_idle)
    else:
        checks, gpu = amd_checks(args.expected, args.require_idle)

    failed = [
        check["name"]
        for check in checks
        if check["required"] and not check["passed"]
    ]
    payload = {
        "schema": "milvus.gpu-doctor.v1",
        "generated_at": utc_now(),
        "backend": args.backend,
        "host": host_metadata(),
        "gpu": gpu,
        "checks": checks,
        "status": "pass" if not failed else "fail",
        "failed_checks": failed,
    }
    write_json(args.output, payload)

    print(f"GPU doctor: {payload['status']}")
    for check in checks:
        state = "PASS" if check["passed"] else "FAIL"
        print(f"[{state}] {check['name']}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
