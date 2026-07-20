# AMD HIP Milvus port — where it lives and how to test (HIP vs CUDA)

## Business value

### Why this exists

Milvus already has a **CPU / Lite** path that is enough for many deployments.
The gap we close is different: teams that need **GPU ANN in Milvus on AMD
hardware** today mostly face a **CUDA-only GPU stack** (or fall back to CPU).
Marketing pages and “should work on Instinct” claims are not a runnable
consumer-GPU path with measured results.

This port delivers a **working HIP GPU line** (hipVS → Knowhere → Milvus) on
**Radeon consumer GPU** (lab: RX 7900 XTX, `gfx1100`), with fair CPU vs GPU
numbers, so the same stack can later scale to **Instinct** without rewriting
the application.

### Who it is for (narrow ICP)

Not “all Milvus users.” Target users are teams for whom CPU/Lite is already
a bottleneck or a strategic dead end:

| Segment | Job to be done | Pain we remove |
|---------|----------------|----------------|
| Startups / SMB with RAG or semantic search | Faster search at target recall on their own box | No affordable AMD GPU path in Milvus; NVIDIA cloud/GPU is costly |
| AMD workstation / small on-prem | Local PoC and production-like GPU ANN | “Docs say GPU” ≠ sealed `GPU_IVF_FLAT` actually works on their card |
| Enterprise pilots before Instinct buy | Validate recall/QPS and ops on AMD first | Dual stack (CUDA PoC → AMD prod) or CPU-only PoC that does not predict GPU |

**CPU/Lite remains the right default** when QPS/latency at the required recall
is fine. GPU on consumer AMD is for when that is no longer true, or when the
org standardizes on AMD GPUs.

### What we proved (evidence, not slogans)

On the same host (`amd-rx7900xtx`), SIFT-1M, IVF `nlist=1024`, `k=10`,
same protocol on both sides (do not mix):

| Protocol | Headline (high-recall `nprobe` 16–32) |
|----------|----------------------------------------|
| Batched harness (one `search()` with 10k queries) | GPU HIP ~**5–7×** CPU; recall@10 matches (~0.93–0.98) |
| VectorDBBench concurrent | GPU HIP ~**2–4×** CPU; GPU holds ~11k QPS as `nprobe` rises |

Plus engineering proof: sealed `GPU_CUVS_IVF_FLAT` build/load/search on
`gfx1100`, including fixes that stock “it should work” paths did not give us
out of the box (e.g. hipVS IVF packer on RDNA3).

### Business value by stakeholder

| Stakeholder | Value |
|-------------|--------|
| Product / DXC | Differentiator: runnable Milvus GPU on AMD with numbers, not only NVIDIA |
| AMD-hardware customers | GPU vector search without CUDA lock-in; same software line toward Instinct |
| AMD / ecosystem | Reference implementation + reproducible benchmarks on consumer RDNA3 |
| Engineering portfolio | Layered port, patches, harness, and fair-compare methodology |

### What this is *not*

- Not a claim that every Milvus user needs a GPU (Lite/CPU stays valid).
- Not “we will conquer AMD” by restating that Instinct support is desired.
- Not a public GitHub dump without a clear why — the why is the gap above
  and the measured HIP port.

### Public release intent

Publish when the story is: **problem → HIP solution on consumer AMD →
measured speed-up / recall → how to reproduce → path to Instinct.**
Feedback from GitHub then validates demand; it does not replace stating
business value first.

---

## Port location (DXC)

The AMD/HIP Layer-3 changes are merged on the Milvus v2.5.4 line:

- <https://github.dxc.com/llmkb-internal/milvus/tree/v2.5.4>

Do **not** expect this port on `amd/main` (or upstream `master`). That tip is a
different Milvus generation; HIP vs CUDA comparisons must use the same
version on both sides (`v2.5.4`).

## Clone Milvus (HIP line)

```bash
git clone https://github.dxc.com/llmkb-internal/milvus.git
cd milvus
git checkout v2.5.4
```

## Knowhere (HIP, required)

GPU indexes use DXC Knowhere 2.5 (Layer-2 HIP already merged):

- <https://github.dxc.com/llmkb-internal/knowhere/tree/2.5>

```bash
git clone https://github.dxc.com/llmkb-internal/knowhere.git
cd knowhere
git checkout 2.5
```

Also need a local hipVS / hipRAFT (ROCm-DS) install used at Milvus build time
(typical lab prefix: `~/rocmds_check_gfx1100/install`).

## Harness (build, smoke, SIFT benchmarks)

Public companion repo:

- <https://github.com/konkolchin/ann-harness-amd>

```bash
git clone https://github.com/konkolchin/ann-harness-amd.git
cd ann-harness-amd
```

Useful entry points (see `docs/` and `scripts/` for details):

- `scripts/build_milvus_layer3.sh` — HIP Milvus build against Knowhere/hipVS
- `scripts/run_milvus_gpu_smoke.sh` — sealed `GPU_IVF_FLAT` smoke (`--flush`)
- `scripts/run_milvus_layer4.sh` — full SIFT-1M nprobe grid (HIP GPU)
- `docs/layer4_run_checklist.md`
- `docs/porting_milvus_gpu_to_amd.tex` — measured HIP vs CPU tables

## Runtime notes (AMD GPU host)

- Pin the discrete GPU if the host also has an iGPU, e.g.:

  ```bash
  export ROCR_VISIBLE_DEVICES=0
  export HIP_VISIBLE_DEVICES=0
  ```

- For sealed GPU search, flush before index/load/search (growing segments
  can stay on CPU `IVF_FLAT_CC`). Layer-3 smoke and Layer-4 scripts do this.
- etcd + minio must be up; use writable rocksmq/runtime paths.

## Fair HIP vs CUDA comparison

1. **HIP side:** build/run Milvus from DXC `v2.5.4` + Knowhere `2.5` + hipVS as above.
2. **CUDA side:** use NVIDIA Milvus GPU at the same release line (Milvus `v2.5.4` /
   matching Knowhere), same dataset and IVF recipe:
   SIFT-1M, `GPU_IVF_FLAT` (or CUDA equivalent), `nlist=1024`, `k=10`,
   `nprobe = 1,4,8,16,32`.
3. Prefer the same client harness (`run_milvus_hdf5.py` / `run_milvus_layer4.sh`
   pattern) on both sides so QPS/recall are comparable.
4. Report recall@10 and QPS/p99 per nprobe; confirm sealed GPU path in logs
   (`GPU_CUVS_IVF_FLAT` / CUDA cuVS equivalents), not growing-path CPU IVF.

Contact / lab reference host: `amd-rx7900xtx` (RX 7900 XTX, gfx1100).
