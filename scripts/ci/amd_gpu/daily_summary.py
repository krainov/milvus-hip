#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .claude_agent import call_claude
    from .common import read_json, utc_now, write_json
except ImportError:
    from claude_agent import call_claude
    from common import read_json, utc_now, write_json


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = Counter(run.get("verdict", "unknown") for run in runs)
    return {
        "schema": "milvus.gpu-daily-facts.v1",
        "generated_at": utc_now(),
        "run_count": len(runs),
        "verdicts": dict(verdicts),
        "latest_run_id": runs[-1].get("run_id") if runs else None,
        "runs": [
            {
                "run_id": run.get("run_id"),
                "commit": run.get("commit"),
                "verdict": run.get("verdict"),
                "generated_at": run.get("generated_at"),
            }
            for run in runs
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic and AI daily summaries")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--facts-output", type=Path, required=True)
    parser.add_argument("--ai-output", type=Path, required=True)
    args = parser.parse_args()

    manifests = sorted(args.runs_dir.glob("*/manifest.json"))
    runs = [read_json(path) for path in manifests[-100:]]
    facts = summarize(runs)
    write_json(args.facts_output, facts)
    analysis = call_claude("daily", facts)
    write_json(args.ai_output, analysis)
    print(f"Daily summary: {len(runs)} runs, AI={analysis['status']}")
    return 1 if analysis["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
