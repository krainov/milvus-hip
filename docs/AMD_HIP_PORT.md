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

Plus engineering proof: sealed `GPU_CUVS_IVF_FLAT` / `GPU_IVF_PQ` build/load/search
on `gfx1100`, and **unfiltered** `GPU_CAGRA` (see [GPU indexes on gfx1100](#gpu-indexes-on-gfx1100-what-works) below).

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
- Not a claim that every Milvus GPU feature (including filtered CAGRA) is
  production-ready on consumer RDNA3 — see restrictions below.

### Public release intent

Publish when the story is: **problem → HIP solution on consumer AMD →
measured speed-up / recall → how to reproduce → path to Instinct.**
Feedback from GitHub then validates demand; it does not replace stating
business value first.

---

## GPU indexes on gfx1100: what works

| Index | Status on RX 7900 XTX (this port) | Notes |
|-------|-----------------------------------|--------|
| `GPU_IVF_FLAT` | **Supported** | Sealed path proven; Layer-4 SIFT recall OK |
| `GPU_IVF_PQ` | **Supported** | Same; use for lower memory / high QPS recipes |
| `GPU_CAGRA` | **Supported for unfiltered search only** | Same scope as AMD Instinct CAGRA demos (no bitset). See usage below. |

### CAGRA — use only without filtering

**Plain English:** CAGRA can search the whole index. It must **not** be used when
some rows must be skipped (deletes, TTL, ACL, scalar / predicate filters).
That “skip some IDs” path is called **filtering** (GPU bitset). On gfx1100 it
is **broken** today (wrong or empty neighbor IDs). AMD’s public CAGRA demos
also show only the unfiltered case.

| Do | Do not |
|----|--------|
| Create `GPU_CAGRA`, load, search with **no** filter expression | Rely on delete / TTL while the CAGRA index is loaded |
| Keep a stable collection (no soft-deletes during search) | Expect metadata / bitset filtering to return correct IDs |
| Prefer IVF indexes if you need deletes or filtered search | Treat CAGRA as “full Milvus feature parity” on this GPU |

#### How to create CAGRA (unfiltered)

Use index type `GPU_CAGRA`. On this GPU, set the graph-build mode that works
here (parameter name in Milvus/Knowhere: `build_algo` = `NN_DESCENT`). Other
build modes may fail on gfx1100.

Example (pymilvus-style params; adjust names to your client API):

```python
index_params = {
    "index_type": "GPU_CAGRA",
    "metric_type": "L2",  # or COSINE / IP as appropriate
    "params": {
        "graph_degree": 32,
        "intermediate_graph_degree": 64,
        "build_algo": "NN_DESCENT",  # required on gfx1100 for a working build
    },
}
# Search: plain vector search only — no filter / expr / bitset.
# Avoid delete() / TTL on this collection while validating CAGRA.
```

Smoke / Layer-4 helpers in the harness:

- `scripts/run_milvus_gpu_cagra_smoke.sh`
- `scripts/run_milvus_layer4_cagra.sh`
- Details: <https://github.com/konkolchin/ann-harness-amd> → `docs/cagra_consumer_followon.md`

Confirm the sealed GPU path in Milvus logs (`GPU_CUVS_CAGRA` / CAGRA), not a
CPU fallback.

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
- `scripts/run_milvus_gpu_cagra_smoke.sh` — sealed unfiltered `GPU_CAGRA` smoke
- `docs/layer4_run_checklist.md`
- `docs/porting_milvus_gpu_to_amd.tex` — measured HIP vs CPU tables
- `docs/cagra_consumer_followon.md` — CAGRA Phase A/B on gfx1100

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
5. For CAGRA compares: unfiltered only on AMD; do not claim filter parity.

Contact / lab reference host: `amd-rx7900xtx` (RX 7900 XTX, gfx1100).

## See also

- [`TEST_FEATURE_NOTES.md`](TEST_FEATURE_NOTES.md) — repository workflow check
  (documentation only, no effect on the HIP port itself).
