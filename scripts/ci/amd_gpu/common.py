from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def median_measurement(values: list[float]) -> float:
    if not values:
        raise ValueError("at least one measurement is required")
    return float(statistics.median(values))


def population_stddev(values: list[float]) -> float:
    if not values:
        raise ValueError("at least one measurement is required")
    return float(statistics.pstdev(values))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_sha256: str | None, label: str) -> str:
    actual_sha256 = sha256(path)
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"{label} SHA256 {actual_sha256} does not match expected "
            f"{expected_sha256}"
        )
    return actual_sha256


def retry_operation(
    operation: Callable[[], T],
    *,
    is_retryable: Callable[[Exception], bool],
    attempts: int = 3,
    delay_s: float = 1.0,
) -> tuple[T, int]:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    for attempt in range(attempts):
        try:
            return operation(), attempt
        except Exception as error:
            if attempt == attempts - 1 or not is_retryable(error):
                raise
            time.sleep(delay_s * (attempt + 1))

    raise AssertionError("retry loop exhausted")


def run_command(
    argv: list[str],
    *,
    timeout: int = 30,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    command_env = os.environ.copy()
    if env:
        command_env.update(env)

    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=command_env,
        )
        return {
            "command": argv,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError:
        return {
            "command": argv,
            "returncode": 127,
            "stdout": "",
            "stderr": f"{argv[0]} was not found",
        }
    except subprocess.TimeoutExpired as error:
        return {
            "command": argv,
            "returncode": 124,
            "stdout": (error.stdout or "").strip(),
            "stderr": (error.stderr or "").strip() or f"timed out after {timeout}s",
        }


def host_metadata() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "kernel": run_command(["uname", "-r"])["stdout"],
    }
