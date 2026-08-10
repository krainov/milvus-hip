#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

try:
    from .common import read_json, sha256, utc_now, write_json
except ImportError:
    from common import read_json, sha256, utc_now, write_json


def markdown_for(manifest: dict[str, Any]) -> str:
    lines = [
        f"# GPU Parity Evidence: {manifest['run_id']}",
        "",
        f"- Generated: `{manifest['generated_at']}`",
        f"- Commit: `{manifest.get('commit', 'unknown')}`",
        f"- Verdict: **{manifest['verdict'].upper()}**",
        "",
        "## Evidence",
        "",
        "| Name | Schema | Status | Gate | SHA256 |",
        "|---|---|---|---|---|",
    ]
    for item in manifest["evidence"]:
        lines.append(
            f"| {item['name']} | {item.get('schema', '')} | "
            f"{item.get('status', '')} | "
            f"{'required' if item.get('gating', True) else 'evidence only'} | "
            f"`{item['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "Correctness determines the verdict. Performance ratios are descriptive.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a checksummed parity evidence pack")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit")
    parser.add_argument("--input", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--non-gating",
        action="append",
        default=[],
        metavar="NAME",
        help="Keep an input in the pack without using its status in the verdict",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence = []
    non_gating = set(args.non_gating)
    input_names = {entry.split("=", 1)[0] for entry in args.input}
    unknown_non_gating = sorted(non_gating - input_names)
    if unknown_non_gating:
        parser.error(
            "--non-gating names are not present in --input: "
            + ", ".join(unknown_non_gating)
        )
    verdict = "pass"
    for entry in args.input:
        name, raw_path = entry.split("=", 1)
        path = Path(raw_path)
        payload = read_json(path)
        item_status = payload.get("status")
        gating = name not in non_gating
        if item_status == "fail" and gating:
            verdict = "fail"
        destination = args.output_dir / path.name
        if path.resolve() != destination.resolve():
            shutil.copy2(path, destination)
        evidence.append(
            {
                "name": name,
                "file": destination.name,
                "sha256": sha256(destination),
                "schema": payload.get("schema"),
                "status": item_status,
                "gating": gating,
            }
        )

    manifest = {
        "schema": "milvus.gpu-evidence-pack.v1",
        "generated_at": utc_now(),
        "run_id": args.run_id,
        "commit": args.commit,
        "verdict": verdict,
        "non_gating_evidence": sorted(non_gating),
        "evidence": evidence,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    (args.output_dir / "README.md").write_text(markdown_for(manifest))
    shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir)
    print(f"Evidence pack: {verdict}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
