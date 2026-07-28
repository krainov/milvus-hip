# AMD GPU parity CI

This directory contains CI-only tooling for validating Milvus AMD GPU support
against the NVIDIA implementation. Production Milvus does not depend on these
scripts.

## Required gates

Every pull request targeting `v2.5.4` runs both GPU jobs. The final verdict
requires all of the following:

- the AMD and NVIDIA jobs use the same pull request commit;
- Milvus is rebuilt from a clean checkout of that exact commit on both runners;
- the running Milvus build advertises that commit in `milvus_build_info`;
- the frozen SIFT-1M dataset checksums match;
- `BRUTEFORCE_C_TEST`, `IVF_FLAT_C_TEST`, `IVF_PQ_C_TEST`, `CAGRA_C_TEST`, and
  `DISTANCE_C_TEST` are all retained in evidence;
- brute-force, IVF_FLAT, IVF_PQ, and distance have the same passing cases on
  hipVS and cuVS. CAGRA remains visible but is non-gating for the current PoC;
- Milvus `GPU_IVF_FLAT` and `GPU_IVF_PQ` recall@10 differs by no more than the
  configured absolute tolerance.

QPS, batch p99, index build time, and sampled peak VRAM are evidence, not
correctness gates. `nprobe=1` is retained in the raw results but excluded from
the correctness verdict because it is too noisy for this comparison.

The benchmark inserts deterministic row IDs before comparing search results to
the exact-KNN neighbors in the dataset. It rejects a stale server even if that
server produces good benchmark numbers.

## GPU isolation

Before Runner Doctor, the workflow acquires exclusive GPU access by gracefully
stopping active GPU compute processes or their owning Docker containers. It
does not remove containers, source trees, build trees, datasets, MinIO, or
etcd. Runner Doctor then requires an idle GPU.

A newly built Milvus is started with run-specific storage roots and
health/provenance checked before product benchmarks. The final `always()`
cleanup stops only the Milvus process or container marked as CI-managed and
leaves dependency containers running.

## Runner contract

The repository expects two repo-scoped self-hosted runners:

| Backend | Required labels |
|---|---|
| AMD RX 7900 XTX | `self-hosted`, `linux`, `x64`, `amd-gpu`, `gfx1100` |
| NVIDIA RTX 4080 | `self-hosted`, `linux`, `x64`, `nvidia-gpu`, `rtx4080` |

Both runners provide these non-secret values in their runner `.env` file:

- `GPU_GTEST_DIR`
- `GPU_LIBRARY_SOURCE_DIR`
- `GPU_BENCH_PYTHON`
- `MILVUS_BENCH_URI`
- `MILVUS_METRICS_URI`
- `MILVUS_BENCH_DATASET`
- `MILVUS_CI_DEPENDENCY_CONTAINERS`

NVIDIA gtests can run in a CUDA build container by also setting:

- `GPU_GTEST_CONTAINER_IMAGE`
- `GPU_GTEST_HOST_SOURCE_DIR`
- `GPU_GTEST_CONTAINER_SOURCE_DIR`

The workflow owns all Milvus image, binary, container, PID, log, and runtime
paths. Runner configuration must not point it at a preinstalled Milvus.

## AI boundary

Claude receives only generated JSON evidence. It has no tools and cannot alter
the verdict. Failure triage and benchmark narration are skipped for pull
requests so that an unmerged workflow never receives `ANTHROPIC_API_KEY`.
Pushes to `v2.5.4` read the key only from the GitHub Actions secret and the
model name from the optional `ANTHROPIC_MODEL` repository variable.

The deterministic verdict, both backend logs, redacted triage, Claude output,
and SHA-256 manifest are uploaded as the run's evidence pack.

GitHub Enterprise Server 3.19 requires the v3 artifact actions; v4 artifact
actions are not compatible with GHES.
