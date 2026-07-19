# AMD HIP Milvus port — where it lives and how to test (HIP vs CUDA)

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
