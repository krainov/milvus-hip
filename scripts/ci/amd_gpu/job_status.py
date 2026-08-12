#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from .common import utc_now, write_json
except ImportError:
    from common import utc_now, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Record GPU job step outcomes")
    parser.add_argument("--backend", choices=("amd", "nvidia"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--step",
        action="append",
        default=[],
        metavar="NAME=OUTCOME",
    )
    args = parser.parse_args()

    steps = {}
    for entry in args.step:
        name, outcome = entry.split("=", 1)
        steps[name] = outcome
    failed = [name for name, outcome in steps.items() if outcome != "success"]
    payload = {
        "schema": "milvus.gpu-job-status.v1",
        "generated_at": utc_now(),
        "backend": args.backend,
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "source_commit": os.environ.get("GITHUB_SHA"),
        "status": "pass" if not failed else "fail",
        "failed_steps": failed,
        "steps": steps,
    }
    write_json(args.output, payload)
    print(f"{args.backend} job status: {payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
