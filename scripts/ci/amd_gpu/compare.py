#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from .common import read_json, utc_now, write_json
except ImportError:
    from common import read_json, utc_now, write_json


def benchmark_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        key: payload.get(key)
        for key in (
            "dataset_id",
            "index_type",
            "nlist",
            "m",
            "nbits",
            "k",
            "dataset_sha256",
        )
    }
    provenance = payload.get("source_provenance", {})
    metadata.update(
        {
            "requested_commit": provenance.get("requested_commit"),
            "server_commit": provenance.get("server_commit"),
        }
    )
    return metadata


def compare_benchmarks(
    amd: dict[str, Any],
    nvidia: dict[str, Any],
    recall_tolerance: float,
    excluded_nprobes: set[int] | None = None,
) -> dict[str, Any]:
    excluded_nprobes = excluded_nprobes or set()
    amd_meta = benchmark_metadata(amd)
    nvidia_meta = benchmark_metadata(nvidia)
    mismatches = {
        key: {"amd": amd_meta[key], "nvidia": nvidia_meta[key]}
        for key in amd_meta
        if amd_meta[key] != nvidia_meta[key]
        and amd_meta[key] is not None
        and nvidia_meta[key] is not None
    }

    amd_rows = {int(row["nprobe"]): row for row in amd["nprobe_results"]}
    nvidia_rows = {int(row["nprobe"]): row for row in nvidia["nprobe_results"]}
    common_nprobes = sorted(set(amd_rows) & set(nvidia_rows))

    rows: list[dict[str, Any]] = []
    for nprobe in common_nprobes:
        amd_row = amd_rows[nprobe]
        nvidia_row = nvidia_rows[nprobe]
        amd_recall = float(amd_row["recall@10"])
        nvidia_recall = float(nvidia_row["recall@10"])
        recall_delta = amd_recall - nvidia_recall
        nvidia_qps = float(nvidia_row["qps"])
        rows.append(
            {
                "nprobe": nprobe,
                "amd": {
                    "qps": float(amd_row["qps"]),
                    "recall_at_10": amd_recall,
                    "p99_batch_ms": amd_row.get(
                        "p99_batch_ms", amd_row.get("p99_ms")
                    ),
                },
                "nvidia": {
                    "qps": nvidia_qps,
                    "recall_at_10": nvidia_recall,
                    "p99_batch_ms": nvidia_row.get(
                        "p99_batch_ms", nvidia_row.get("p99_ms")
                    ),
                },
                "recall_delta": recall_delta,
                "recall_pass": abs(recall_delta) <= recall_tolerance,
                "correctness_gate_applied": nprobe not in excluded_nprobes,
                "qps_ratio_amd_to_nvidia": (
                    float(amd_row["qps"]) / nvidia_qps if nvidia_qps else None
                ),
            }
        )

    failed_rows = [
        row["nprobe"]
        for row in rows
        if row["correctness_gate_applied"] and not row["recall_pass"]
    ]
    input_failures = [
        backend
        for backend, payload in (("amd", amd), ("nvidia", nvidia))
        if payload.get("status") == "fail"
    ]
    status = (
        "pass"
        if rows and not mismatches and not failed_rows and not input_failures
        else "fail"
    )
    return {
        "schema": "milvus.gpu-benchmark-parity.v1",
        "generated_at": utc_now(),
        "kind": "benchmark",
        "status": status,
        "correctness_gate": {
            "metric": "absolute recall@10 delta",
            "tolerance": recall_tolerance,
            "excluded_nprobes": sorted(excluded_nprobes),
            "failed_nprobes": failed_rows,
        },
        "input_failures": input_failures,
        "metadata": amd_meta,
        "metadata_mismatches": mismatches,
        "rows": rows,
    }


def flatten_gtests(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for suite in payload.get("suite", []):
        binary = suite.get("binary", "unknown")
        for case in suite.get("cases", []):
            cases[f"{binary}::{case['name']}"] = case
    return cases


def geometric_mean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0]
    if not positive:
        return None
    return math.exp(statistics.fmean(math.log(value) for value in positive))


def compare_gtests(
    amd: dict[str, Any],
    nvidia: dict[str, Any],
    excluded_binaries: set[str] | None = None,
) -> dict[str, Any]:
    excluded_binaries = excluded_binaries or set()
    amd_cases = flatten_gtests(amd)
    nvidia_cases = flatten_gtests(nvidia)
    shared_names = sorted(set(amd_cases) & set(nvidia_cases))
    missing_on_amd = sorted(set(nvidia_cases) - set(amd_cases))
    missing_on_nvidia = sorted(set(amd_cases) - set(nvidia_cases))

    rows: list[dict[str, Any]] = []
    timing_ratios: list[float] = []
    for name in shared_names:
        amd_case = amd_cases[name]
        nvidia_case = nvidia_cases[name]
        amd_ms = float(amd_case.get("ms", 0))
        nvidia_ms = float(nvidia_case.get("ms", 0))
        ratio = amd_ms / nvidia_ms if nvidia_ms else None
        if ratio:
            timing_ratios.append(ratio)
        rows.append(
            {
                "name": name,
                "correctness_gate_applied": (
                    name.split("::", 1)[0] not in excluded_binaries
                ),
                "amd_status": amd_case.get("status"),
                "nvidia_status": nvidia_case.get("status"),
                "amd_ms": amd_ms,
                "nvidia_ms": nvidia_ms,
                "timing_ratio_amd_to_nvidia": ratio,
                "correctness_pass": (
                    amd_case.get("status") == "PASS"
                    and nvidia_case.get("status") == "PASS"
                ),
            }
        )

    failed_cases = [
        row["name"]
        for row in rows
        if row["correctness_gate_applied"] and not row["correctness_pass"]
    ]
    gating_missing_on_amd = [
        name
        for name in missing_on_amd
        if name.split("::", 1)[0] not in excluded_binaries
    ]
    gating_missing_on_nvidia = [
        name
        for name in missing_on_nvidia
        if name.split("::", 1)[0] not in excluded_binaries
    ]
    status = (
        "pass"
        if rows
        and not gating_missing_on_amd
        and not gating_missing_on_nvidia
        and not failed_cases
        else "fail"
    )
    return {
        "schema": "milvus.gpu-gtest-parity.v1",
        "generated_at": utc_now(),
        "kind": "gtest",
        "status": status,
        "summary": {
            "shared_cases": len(rows),
            "failed_cases": len(failed_cases),
            "missing_on_amd": len(missing_on_amd),
            "missing_on_nvidia": len(missing_on_nvidia),
            "timing_ratio_geomean_amd_to_nvidia": geometric_mean(timing_ratios),
        },
        "excluded_binaries": sorted(excluded_binaries),
        "missing_on_amd": missing_on_amd,
        "missing_on_nvidia": missing_on_nvidia,
        "failed_cases": failed_cases,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare AMD and NVIDIA evidence")
    parser.add_argument("--kind", choices=("benchmark", "gtest"), required=True)
    parser.add_argument("--amd", type=Path, required=True)
    parser.add_argument("--nvidia", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recall-tolerance", type=float, default=0.005)
    parser.add_argument(
        "--exclude-nprobe",
        type=int,
        action="append",
        default=[],
        help="Keep this nprobe in evidence but exclude it from correctness verdict",
    )
    parser.add_argument(
        "--exclude-binary",
        action="append",
        default=[],
        help="Keep this gtest binary in evidence but exclude it from the current gate",
    )
    parser.add_argument("--fail-on-correctness", action="store_true")
    args = parser.parse_args()

    amd = read_json(args.amd)
    nvidia = read_json(args.nvidia)
    if args.kind == "benchmark":
        result = compare_benchmarks(
            amd,
            nvidia,
            args.recall_tolerance,
            set(args.exclude_nprobe),
        )
    else:
        result = compare_gtests(amd, nvidia, set(args.exclude_binary))

    write_json(args.output, result)
    print(f"{args.kind} parity: {result['status']}")
    if args.fail_on_correctness and result["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
