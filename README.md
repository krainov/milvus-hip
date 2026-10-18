# Milvus on AMD GPUs (HIP) — tech preview

**Public fork of Milvus `v2.5.4` with AMD HIP / ROCm GPU ANN support**
(consumer RDNA3 validated in lab: RX 7900 XTX, `gfx1100`).

This is **not** upstream [milvus-io/milvus](https://github.com/milvus-io/milvus)
`master`, and it is **not** a claim of full NVIDIA GPU feature or speed parity.

| Piece | Public repo | Branch / pin |
|-------|-------------|--------------|
| **Milvus (this repo)** | https://github.com/konkolchin/milvus-hip | `v2.5.4` |
| **Knowhere (required)** | https://github.com/konkolchin/knowhere-hip | `2.5` |
| **Bench / build harness** | https://github.com/konkolchin/ann-harness-amd | `master` |
| **hipVS / hipRAFT** | Upstream [ROCm-DS/hipVS](https://github.com/ROCm-DS/hipVS) + [hipRaft](https://github.com/ROCm-DS/hipRaft) | e.g. `release/rocmds-25.10` + gfx1100 IVF packer patch (see harness) |


## Client vs this server

This repo is the **Milvus server**, not a pip package. GPU search runs in the
process you build from this tree (Knowhere → hipVS → HIP). The Python API is
stock [pymilvus](https://github.com/milvus-io/pymilvus). It does not know about
HIP or this fork. It only talks to whatever Milvus is listening
(default `127.0.0.1:19530`).

```bash
pip install pymilvus
```

`19530` is the default gRPC port of a running Milvus standalone.
Build this tree first (see below), **or** run the published ROCm image
([`deployments/docker/standalone-rocm`](deployments/docker/standalone-rocm/README.md)).
The official `milvusdb/milvus` GPU image is CUDA. It does not give GPU search
on a Radeon. Stop any leftover `milvus-standalone` container first — it can
load the wrong `libcuvs`.

**Docker image** `ghcr.io/krainov/milvus-hip:v2.5.4-rocm`: Milvus v2.5.4,
RX 7900 XTX (`gfx1100`). **`GPU_IVF_FLAT` verified** (sealed `GPU_CUVS_IVF_FLAT`,
slice-exact recall@10 = 1.0 at nprobe 128). **`GPU_IVF_PQ`, `GPU_CAGRA`, and
Instinct are not claimed** for this image.

When the server is up:

```python
from pymilvus import connections
connections.connect(host="127.0.0.1", port="19530")
```

```bash
docker pull ghcr.io/krainov/milvus-hip:v2.5.4-rocm
docker rm -f milvus-standalone 2>/dev/null || true
cd deployments/docker/standalone-rocm
export MILVUS_HIP_IMAGE=ghcr.io/krainov/milvus-hip:v2.5.4-rocm
bash up.sh
bash smoke.sh
```

---

## Indexes

| Index | Status | Notes |
|-------|--------|--------|
| `GPU_IVF_FLAT` | Supported | Validated on RX 7900 XTX (`gfx1100`) |
| `GPU_IVF_PQ` | Supported | Validated on RX 7900 XTX (`gfx1100`) |
| `GPU_CAGRA` | Unfiltered search only | Use `build_algo=NN_DESCENT` on the validated Radeon path. hipVS CAGRA search with a bitset (filters, deletes, TTL) is not supported in this port. Prefer IVF if you need that. |

### Which GPU

This tree is a HIP port of Milvus. The same software line is meant for AMD GPUs.
We validated it on an RX 7900 XTX (`gfx1100`). We have not tested Instinct.

The IVF packer patch is for **wavefront 32** only. Instinct GPUs use wavefront 64.
Do not apply that patch there.

| GPU | Wavefront | What to do |
|-----|-----------|------------|
| Radeon RX 7900 XTX (`gfx1100`) | 32 (this port) | Apply the harness IVF packer patch. This is the validated path. |
| Instinct MI200 / MI300 (`gfx90a` / `gfx942`) | 64 | Do not apply our hipVS packer patch. hipVS already targets Instinct. This Milvus tree is untested on Instinct. |

### CAGRA — unfiltered only

Use CAGRA when you search the **whole** collection. Do **not** rely on it when
some IDs must be skipped (deletes, TTL, ACL, scalar filters). Prefer IVF indexes
if you need that.

```python
index_params = {
    "index_type": "GPU_CAGRA",
    "metric_type": "L2",
    "params": {
        "graph_degree": 32,
        "intermediate_graph_degree": 64,
        "build_algo": "NN_DESCENT",  # required on the validated gfx1100 path
    },
}
# Search with no filter expression. Avoid delete()/TTL while validating CAGRA.
```

---

## Clone

```bash
# Milvus HIP line
git clone -b v2.5.4 https://github.com/konkolchin/milvus-hip.git milvus
cd milvus

# Knowhere HIP line (sibling clone; build pins this repo @ 2.5)
git clone -b 2.5 https://github.com/konkolchin/knowhere-hip.git ../knowhere

# Harness (smoke + Layer-4 benches)
git clone https://github.com/konkolchin/ann-harness-amd.git ../ann-harness-amd
```

Do **not** mix this `v2.5.4` HIP line with a random newer Milvus tip when
comparing to NVIDIA — use the **same** Milvus generation on both GPUs.

---

## Build dependencies (short)

1. **ROCm** on the AMD host (lab used ROCm 7.0.2).
2. Build/install **hipRaft** then **hipVS** from ROCm-DS into a prefix
   (e.g. `~/rocmds_check_gfx1100/install`). On `gfx1100`, apply the IVF
   packer warp32 patch from the harness (`patches/hipvs/0002-…`) before
   install — see [ann-harness-amd](https://github.com/konkolchin/ann-harness-amd).
3. Build this Milvus tree against **Knowhere `2.5`** from
   `konkolchin/knowhere-hip` and `HIPVS_PREFIX=<your install>`.

Harness entry points:

```bash
cd ../ann-harness-amd
# HIP Milvus build against Knowhere + hipVS (see script comments for env)
bash scripts/build_milvus_layer3.sh

# Sealed GPU_IVF_FLAT smoke (Milvus already on :19530)
bash scripts/run_milvus_gpu_smoke.sh
```

Point any Knowhere Git/CMake pin at:

```text
https://github.com/konkolchin/knowhere-hip.git
branch / tag: 2.5
```

If your tree uses CMake overrides, typical pattern:

```bash
# Example only — match names used in your build scripts
export MILVUS_KNOWHERE_GIT_REPOSITORY=https://github.com/konkolchin/knowhere-hip.git
export MILVUS_KNOWHERE_VERSION=2.5
# or: -DMILVUS_KNOWHERE_SOURCE_DIR=/path/to/knowhere-hip
```

---

## Runtime notes (AMD)

```bash
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
```

Flush before index/load/search so QueryNode uses the **sealed GPU** index
(growing segments can stay on CPU `IVF_FLAT_CC`). Harness smoke/Layer-4
scripts already do this.

---

## Benchmarks / fair HIP vs CUDA

Use the harness so both GPUs share the same client protocol:

- https://github.com/konkolchin/ann-harness-amd  
  (`run_milvus_layer4_*.sh`, median: `SEARCH_WARMUP=5 SEARCH_RUNS=10`)

Honest expectations from lab (peer GPUs, not identical silicon):

- **IVF_FLAT:** strong / usable on AMD  
- **IVF_PQ:** correct recall; often behind CUDA on product and library paths  
- **CAGRA (unfiltered):** works  
- **CAGRA (filtered):** out of scope (hipVS bitset path)

---

## Built with AI agents

This port was developed with heavy use of **AI coding agents** as part of how
our team delivers CUDA→HIP work — notably **[Cursor](https://cursor.com)**
(with agent assistance), alongside other tools used by teammates (e.g. Claude).
Human engineers still own design, measurement, and what we claim publicly.

---

## License / affiliation

- Upstream Milvus / Knowhere licensing applies (Apache License 2.0) — keep
  `LICENSE` / notices from the trees you push.
- This public tech preview is maintained on a **personal** GitHub account for
  visibility. It is an AMD HIP port of the `v2.5.4` line, not an official
  Zilliz release.

## Status

**Tech preview.** API and indexes above are what we stand behind on consumer
RDNA3 today. Contributions and reproduce reports welcome via Issues on this
repo and on the harness.
