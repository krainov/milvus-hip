#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import socket
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from pymilvus import DataType, MilvusClient

try:
    from .common import retry_operation, sha256, utc_now, verify_sha256, write_json
except ImportError:
    from common import retry_operation, sha256, utc_now, verify_sha256, write_json


TRANSIENT_MILVUS_ERRORS = (
    "node not match",
    "statuscode.unavailable",
    "service unavailable",
    "connection refused",
)


def batched(array: np.ndarray, batch_size: int):
    for offset in range(0, len(array), batch_size):
        yield offset, array[offset : offset + batch_size]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def recall_at_k(actual: list[list[int]], expected: np.ndarray, k: int) -> float:
    matched = 0
    for actual_ids, expected_ids in zip(actual, expected):
        matched += len(set(actual_ids[:k]) & set(int(value) for value in expected_ids[:k]))
    return matched / (len(expected) * k)


def search_ids(result: Any) -> list[list[int]]:
    return [[int(hit["id"]) for hit in hits] for hits in result]


def is_transient_milvus_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in TRANSIENT_MILVUS_ERRORS)


def amd_vram_used_mib() -> float | None:
    devices = []
    for total_path in Path("/sys/class/drm").glob("card*/device/mem_info_vram_total"):
        try:
            total = int(total_path.read_text())
            used = int((total_path.parent / "mem_info_vram_used").read_text())
            devices.append((total, used))
        except (OSError, ValueError):
            continue
    if not devices:
        return None
    _, used = max(devices)
    return used / (1024 * 1024)


def nvidia_vram_used_mib() -> float | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    try:
        return max(float(line.strip()) for line in result.stdout.splitlines() if line.strip())
    except ValueError:
        return None


def server_build_info(metrics_uri: str | None) -> dict[str, str | None]:
    if not metrics_uri:
        return {"commit": None, "version": None, "built": None}
    try:
        with urllib.request.urlopen(metrics_uri, timeout=10) as response:
            metrics = response.read().decode(errors="replace")
    except (OSError, TimeoutError):
        return {"commit": None, "version": None, "built": None}

    line = next(
        (line for line in metrics.splitlines() if line.startswith("milvus_build_info{")),
        "",
    )
    labels = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', line))
    return {
        "commit": labels.get("git_commit"),
        "version": labels.get("version"),
        "built": labels.get("built"),
    }


class GpuMemorySampler:
    def __init__(self, backend: str, interval_s: float = 0.2) -> None:
        self.backend = backend
        self.interval_s = interval_s
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def _sample_loop(self) -> None:
        sampler = amd_vram_used_mib if self.backend == "amd" else nvidia_vram_used_mib
        while not self._stop.is_set():
            try:
                value = sampler()
                if value is not None:
                    self.samples.append(value)
            except (OSError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> dict[str, float | int | None]:
        self._stop.set()
        self._thread.join(timeout=5)
        return {
            "sample_count": len(self.samples),
            "min_used_mib": min(self.samples) if self.samples else None,
            "mean_used_mib": statistics.fmean(self.samples) if self.samples else None,
            "peak_used_mib": max(self.samples) if self.samples else None,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milvus exact-KNN parity benchmark")
    parser.add_argument(
        "--uri",
        default=os.environ.get("MILVUS_BENCH_URI", "http://127.0.0.1:19530"),
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--backend", choices=("amd", "nvidia"), required=True)
    parser.add_argument(
        "--index-type",
        choices=("GPU_IVF_FLAT", "GPU_IVF_PQ"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection")
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nprobe", type=int, action="append", dest="nprobes")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--insert-batch", type=int, default=10_000)
    parser.add_argument("--query-batch", type=int, default=100)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-query-rows", type=int)
    parser.add_argument("--keep-collection", action="store_true")
    parser.add_argument(
        "--server-source-commit",
        default=os.environ.get("MILVUS_SERVER_SOURCE_COMMIT"),
    )
    parser.add_argument(
        "--metrics-uri",
        default=os.environ.get("MILVUS_METRICS_URI", "http://127.0.0.1:9091/metrics"),
    )
    parser.add_argument("--require-source-match", action="store_true")
    return parser.parse_args()


def benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    dataset_sha256 = verify_sha256(
        args.dataset,
        args.expected_dataset_sha256,
        "dataset",
    )
    dataset_id = args.dataset_id or args.dataset.stem
    requested_commit = os.environ.get("GITHUB_SHA")
    build_info = server_build_info(args.metrics_uri)
    server_commit = args.server_source_commit or build_info["commit"]
    source_matches = bool(
        requested_commit
        and server_commit
        and requested_commit.startswith(server_commit)
    )
    provenance_status = (
        "pass"
        if source_matches
        else ("fail" if args.require_source_match else "unverified")
    )
    nprobes = args.nprobes or [4, 8, 16, 32]

    with h5py.File(args.dataset, "r") as dataset:
        train = np.asarray(dataset["train"], dtype=np.float32)
        queries = np.asarray(dataset["test"], dtype=np.float32)
        neighbors = np.asarray(dataset["neighbors"], dtype=np.int64)
    if args.max_train_rows:
        train = train[: args.max_train_rows]
    if args.max_query_rows:
        queries = queries[: args.max_query_rows]
        neighbors = neighbors[: args.max_query_rows]

    collection = args.collection or (
        f"ci_{args.backend}_{args.index_type.lower()}_{int(time.time())}"
    )
    client = MilvusClient(uri=args.uri)
    server_version = client.get_server_version()
    if client.has_collection(collection):
        client.drop_collection(collection)

    timings: dict[str, float] = {}
    operation_retries: dict[str, int] = {}
    results = []
    memory = GpuMemorySampler(args.backend)
    memory.start()
    try:
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=int(train.shape[1]))

        def create_collection() -> None:
            if client.has_collection(collection):
                client.drop_collection(collection)
            client.create_collection(collection_name=collection, schema=schema)

        _, operation_retries["create_collection"] = retry_operation(
            create_collection,
            is_retryable=is_transient_milvus_error,
            attempts=3,
            delay_s=5,
        )

        started = time.perf_counter()
        for offset, batch in batched(train, args.insert_batch):
            client.insert(
                collection_name=collection,
                data=[
                    {"id": offset + index, "vec": row.tolist()}
                    for index, row in enumerate(batch)
                ],
            )
        timings["insert"] = time.perf_counter() - started

        started = time.perf_counter()
        client.flush(collection_name=collection)
        timings["flush"] = time.perf_counter() - started

        index_params = client.prepare_index_params()
        params = {"nlist": args.nlist}
        if args.index_type == "GPU_IVF_PQ":
            params.update({"m": args.m, "nbits": args.nbits})
        index_params.add_index(
            field_name="vec",
            index_type=args.index_type,
            metric_type="L2",
            params=params,
        )
        started = time.perf_counter()
        client.create_index(collection_name=collection, index_params=index_params)
        timings["index_build"] = time.perf_counter() - started

        started = time.perf_counter()
        client.load_collection(collection_name=collection)
        timings["load"] = time.perf_counter() - started

        warmup_size = min(len(queries), args.query_batch)
        if warmup_size:
            client.search(
                collection_name=collection,
                data=queries[:warmup_size].tolist(),
                anns_field="vec",
                limit=args.k,
                search_params={
                    "metric_type": "L2",
                    "params": {"nprobe": nprobes[0]},
                },
                output_fields=[],
            )

        for nprobe in nprobes:
            found: list[list[int]] = []
            latencies: list[float] = []
            started_all = time.perf_counter()
            for _, batch in batched(queries, args.query_batch):
                started = time.perf_counter()
                response = client.search(
                    collection_name=collection,
                    data=batch.tolist(),
                    anns_field="vec",
                    limit=args.k,
                    search_params={
                        "metric_type": "L2",
                        "params": {"nprobe": nprobe},
                    },
                    output_fields=[],
                )
                latencies.append((time.perf_counter() - started) * 1000)
                found.extend(search_ids(response))
            elapsed = time.perf_counter() - started_all
            results.append(
                {
                    "nprobe": nprobe,
                    "qps": len(queries) / elapsed,
                    "recall@10": recall_at_k(found, neighbors, args.k),
                    "p50_batch_ms": statistics.median(latencies),
                    "p99_batch_ms": percentile(latencies, 0.99),
                    "query_batch_size": args.query_batch,
                }
            )
    finally:
        vram = memory.finish()
        if not args.keep_collection and client.has_collection(collection):
            client.drop_collection(collection)

    payload = {
        "schema": "milvus.gpu-product-benchmark.v2",
        "generated_at": utc_now(),
        "status": "fail" if provenance_status == "fail" else "pass",
        "backend": args.backend,
        "host": socket.gethostname(),
        "dataset_id": dataset_id,
        "uri": args.uri,
        "server_version": server_version,
        "server_build": build_info,
        "source_provenance": {
            "status": provenance_status,
            "requested_commit": requested_commit,
            "server_commit": server_commit,
        },
        "collection": collection,
        "index_type": args.index_type,
        "nlist": args.nlist,
        "m": args.m if args.index_type == "GPU_IVF_PQ" else None,
        "nbits": args.nbits if args.index_type == "GPU_IVF_PQ" else None,
        "k": args.k,
        "nprobes": nprobes,
        "xb_shape": list(train.shape),
        "xq_shape": list(queries.shape),
        "dataset_sha256": dataset_sha256,
        "operation_retries": operation_retries,
        "timings_s": timings,
        "vram": vram,
        "nprobe_results": results,
    }
    return payload, 1 if provenance_status == "fail" else 0


def main() -> int:
    args = parse_args()
    try:
        payload, returncode = benchmark(args)
    except Exception as error:
        payload = {
            "schema": "milvus.gpu-product-benchmark.v2",
            "generated_at": utc_now(),
            "status": "fail",
            "backend": args.backend,
            "dataset_id": args.dataset_id or args.dataset.stem,
            "index_type": args.index_type,
            "dataset_sha256": sha256(args.dataset) if args.dataset.is_file() else None,
            "source_provenance": {
                "status": "fail" if args.require_source_match else "unverified",
                "requested_commit": os.environ.get("GITHUB_SHA"),
                "server_commit": args.server_source_commit,
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "nprobe_results": [],
        }
        returncode = 1
    write_json(args.output, payload)
    print(
        f"{args.backend} {args.index_type}: "
        f"{len(payload.get('nprobe_results', []))} nprobe results; "
        f"status={payload['status']}"
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
