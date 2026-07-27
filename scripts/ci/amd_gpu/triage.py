#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

try:
    from .common import utc_now, write_json
except ImportError:
    from common import utc_now, write_json


SECRET_PATTERNS = [
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)(authorization:\s*(?:bearer|token)\s+)[^\s]+"),
    re.compile(r"(?i)((?:password|passwd|api[_-]?key|token)\s*[=:]\s*)[^\s]+"),
]

SIGNATURES = [
    ("runner", "runner_offline", re.compile(r"no runner matching|runner.*offline", re.I)),
    ("infrastructure", "disk_full", re.compile(r"no space left on device", re.I)),
    (
        "infrastructure",
        "docker_failure",
        re.compile(
            r"(?:docker[^\n]{0,160}(?:daemon|timeout|failed|error))|"
            r"(?:(?:failed|error)[^\n]{0,160}docker)",
            re.I,
        ),
    ),
    ("infrastructure", "gpu_busy", re.compile(r"active compute processes|device.*busy", re.I)),
    (
        "compiler",
        "hip_compile",
        re.compile(
            r"(?:hipcc|clang|offload-arch)[^\n]{0,200}"
            r"(?:error:|failed|unsupported|not found)",
            re.I,
        ),
    ),
    (
        "compiler",
        "cuda_compile",
        re.compile(
            r"(?:nvcc|cuda|gpu architecture)[^\n]{0,200}"
            r"(?:error:|failed|unsupported|not found)",
            re.I,
        ),
    ),
    (
        "infrastructure",
        "milvus_node_mismatch",
        re.compile(r"node not match\[expectedNodeID=.*?\]\[actualNodeID=.*?\]", re.I),
    ),
    ("correctness", "gtest_failure", re.compile(r"\[\s*FAILED\s*\]|assertion.*failed", re.I)),
    ("correctness", "recall_gate", re.compile(r"recall.*(?:delta|tolerance|failed)", re.I)),
    (
        "benchmark",
        "benchmark_failure",
        re.compile(r"benchmark[^\n]{0,160}(?:failed|error|exception)", re.I),
    ),
]


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", text)
    return text


def classify(text: str) -> list[dict[str, Any]]:
    matches = []
    for category, signature, pattern in SIGNATURES:
        found = pattern.search(text)
        if found:
            matches.append(
                {
                    "category": category,
                    "signature": signature,
                    "evidence": found.group(0)[:240],
                }
            )
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify and redact a CI failure log")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=100_000)
    args = parser.parse_args()

    raw = args.log.read_text(errors="replace")
    excerpt = redact(raw[-args.max_chars :])
    signatures = classify(excerpt)
    payload = {
        "schema": "milvus.gpu-failure-triage.v1",
        "generated_at": utc_now(),
        "status": "classified" if signatures else "unknown",
        "primary_category": signatures[0]["category"] if signatures else "unknown",
        "signatures": signatures,
        "log_excerpt": excerpt,
    }
    write_json(args.output, payload)
    print(f"Failure category: {payload['primary_category']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
