#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PATTERNS = {
    "github_pat": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "credential_assignment": re.compile(
        r"(?i)\b(?:password|passwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*"
        r"['\"]?(?P<value>[A-Za-z0-9/+_.:-]{8,})"
    ),
}

ALLOWLIST_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".gz",
    ".so",
    ".a",
}


def tracked_files(root: Path, base: str | None) -> list[Path]:
    command = ["git", "ls-files", "-z"]
    if base:
        command = ["git", "diff", "--name-only", "-z", f"{base}...HEAD"]
    result = subprocess.run(
        command,
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [
        root / entry.decode()
        for entry in result.stdout.split(b"\0")
        if entry
    ]


def scan_file(path: Path, root: Path) -> list[str]:
    if path.suffix.lower() in ALLOWLIST_SUFFIXES or not path.is_file():
        return []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []

    findings: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if "pragma: allowlist secret" in line:
            continue
        for pattern_name, pattern in PATTERNS.items():
            match = pattern.search(line)
            if (
                match
                and pattern_name == "credential_assignment"
                and match.group("value").startswith(
                    ("os.environ.", "os.getenv", "environ.", "secrets.", "vars.")
                )
            ):
                continue
            if match:
                relative = path.relative_to(root)
                findings.append(f"{relative}:{line_number}: {pattern_name}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan tracked files for credentials")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--base", help="Scan only files changed since this commit")
    args = parser.parse_args()

    root = args.root.resolve()
    findings = [
        finding
        for path in tracked_files(root, args.base)
        for finding in scan_file(path, root)
    ]
    if findings:
        print("Potential credentials found in tracked files:", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        return 1

    print("No credential patterns found in tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
