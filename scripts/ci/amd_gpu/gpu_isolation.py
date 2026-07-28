#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
from pathlib import Path
from typing import Any

try:
    from .common import run_command, utc_now, write_json
except ImportError:
    from common import run_command, utc_now, write_json


CONTAINER_ID = re.compile(r"(?:docker[-/]|/)([0-9a-f]{64})(?:\.scope|/|$)")


def parse_nvidia_processes(text: str) -> list[dict[str, Any]]:
    processes = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if not fields or not fields[0].isdigit():
            continue
        processes.append(
            {
                "pid": int(fields[0]),
                "name": fields[1] if len(fields) > 1 else "",
                "memory_mib": fields[2] if len(fields) > 2 else "",
            }
        )
    return processes


def parse_amd_processes(text: str) -> list[dict[str, Any]]:
    try:
        system = json.loads(text or "{}").get("system", {})
    except json.JSONDecodeError:
        return []

    processes = []
    for key, details in system.items():
        match = re.fullmatch(r"PID(\d+)", key)
        if not match:
            continue
        fields = [field.strip() for field in str(details).split(",")]
        processes.append(
            {
                "pid": int(match.group(1)),
                "name": fields[0] if fields else "",
                "memory_bytes": fields[2] if len(fields) > 2 else "",
            }
        )
    return processes


def list_gpu_processes(backend: str) -> list[dict[str, Any]]:
    if backend == "nvidia":
        result = run_command(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        if result["returncode"] != 0:
            raise RuntimeError(result["stderr"] or "nvidia-smi failed")
        return parse_nvidia_processes(result["stdout"])

    result = run_command(["rocm-smi", "--showpids", "--json"])
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"] or "rocm-smi failed")
    return parse_amd_processes(result["stdout"])


def protected_pids() -> set[int]:
    protected = {os.getpid()}
    pid = os.getppid()
    while pid > 1 and pid not in protected:
        protected.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return protected


def container_id_for_pid(pid: int) -> str | None:
    try:
        cgroup = Path(f"/proc/{pid}/cgroup").read_text()
    except OSError:
        return None
    match = CONTAINER_ID.search(cgroup)
    return match.group(1) if match else None


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_gpu_idle(backend: str, timeout: int) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    remaining = list_gpu_processes(backend)
    while remaining and time.monotonic() < deadline:
        time.sleep(1)
        remaining = list_gpu_processes(backend)
    return remaining


def stop_processes(
    backend: str, processes: list[dict[str, Any]], grace_seconds: int
) -> list[dict[str, Any]]:
    actions = []
    protected = protected_pids()
    stopped_containers: set[str] = set()

    for process in processes:
        pid = int(process["pid"])
        if pid in protected:
            raise RuntimeError(f"refusing to terminate CI ancestor PID {pid}")

        container_id = container_id_for_pid(pid)
        if container_id:
            if container_id in stopped_containers:
                continue
            result = run_command(
                ["docker", "stop", "--time", str(grace_seconds), container_id],
                timeout=grace_seconds + 15,
            )
            if result["returncode"] != 0:
                raise RuntimeError(
                    result["stderr"] or f"cannot stop container {container_id}"
                )
            stopped_containers.add(container_id)
            actions.append(
                {
                    "action": "docker-stop",
                    "container_id": container_id,
                    "pid": pid,
                    "process": process.get("name", ""),
                }
            )
            continue

        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        actions.append(
            {
                "action": "sigterm",
                "pid": pid,
                "process": process.get("name", ""),
            }
        )

    remaining = wait_for_gpu_idle(backend, grace_seconds)
    for process in remaining:
        pid = int(process["pid"])
        container_id = container_id_for_pid(pid)
        if container_id:
            result = run_command(["docker", "kill", container_id])
            if result["returncode"] != 0:
                raise RuntimeError(
                    result["stderr"] or f"cannot kill container {container_id}"
                )
            actions.append(
                {
                    "action": "docker-kill",
                    "container_id": container_id,
                    "pid": pid,
                    "process": process.get("name", ""),
                }
            )
        elif pid_exists(pid):
            os.kill(pid, signal.SIGKILL)
            actions.append(
                {
                    "action": "sigkill",
                    "pid": pid,
                    "process": process.get("name", ""),
                }
            )

    return actions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire exclusive use of a self-hosted CI GPU"
    )
    parser.add_argument("--backend", choices=("amd", "nvidia"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grace-seconds", type=int, default=30)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "schema": "milvus.gpu-isolation.v1",
        "generated_at": utc_now(),
        "backend": args.backend,
        "status": "fail",
        "initial_processes": [],
        "actions": [],
        "remaining_processes": [],
    }
    try:
        initial = list_gpu_processes(args.backend)
        payload["initial_processes"] = initial
        payload["actions"] = stop_processes(
            args.backend, initial, args.grace_seconds
        )
        remaining = wait_for_gpu_idle(args.backend, 5)
        payload["remaining_processes"] = remaining
        if remaining:
            raise RuntimeError("GPU still has active compute processes")
        payload["status"] = "pass"
        print(
            f"GPU isolation: pass; performed {len(payload['actions'])} stop action(s)"
        )
    except Exception as error:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        print(f"GPU isolation: fail; {error}")
    write_json(args.output, payload)
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
