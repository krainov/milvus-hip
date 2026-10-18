# Milvus HIP standalone (ROCm Docker)

Tech preview image for Milvus **v2.5.4** GPU ANN on AMD **HIP/ROCm**.

Official `milvusdb/milvus` GPU tags are **CUDA**. They do not give GPU search
on a Radeon. Stock [pymilvus](https://github.com/milvus-io/pymilvus) talks to
port **19530**. There is no HIP build of pymilvus.

## What this image claims

Image: `ghcr.io/konkolchin/milvus-hip:v2.5.4-rocm`

| Claim | Status |
|-------|--------|
| Milvus | v2.5.4 HIP pack (July lab binary) |
| GPU | RX 7900 XTX (`gfx1100`), ROCm 7.0.2 |
| `GPU_IVF_FLAT` | **Verified** — sealed `GPU_CUVS_IVF_FLAT` build/load/search; slice-exact recall@10 = 1.0 at nprobe 128 (50k×128 SIFT) |
| `GPU_IVF_PQ` | **Not claimed** for this image (host port has lab benches; Docker smoke not run) |
| `GPU_CAGRA` | **Not claimed** |
| Instinct | **Not claimed** (untested). Do not apply the harness wave32 IVF packer on wave64 Instinct |

## Pull and run (AMD host with `/dev/kfd`)

```bash
docker pull ghcr.io/konkolchin/milvus-hip:v2.5.4-rocm
docker rm -f milvus-standalone milvus-hip-standalone 2>/dev/null || true

cd deployments/docker/standalone-rocm
export MILVUS_HIP_IMAGE=ghcr.io/konkolchin/milvus-hip:v2.5.4-rocm
bash up.sh
bash smoke.sh
```

`up.sh` starts etcd/MinIO and `docker run` with `--device /dev/kfd`, `--ipc=host`.
Ubuntu apt `docker-compose` 1.x hits `KeyError: ContainerConfig` on this image.

```python
from pymilvus import connections
connections.connect(host="127.0.0.1", port="19530")
```

## Rebuild from a lab prefix (packers)

On the AMD box, after Layer-3 (`ann-harness-amd` `build_milvus_layer3.sh`):

```bash
docker rm -f milvus-standalone 2>/dev/null || true
cd deployments/docker/standalone-rocm
export MILVUS_DIR=$HOME/rocmds_check_gfx1100/milvus
export HIPVS_PREFIX=$HOME/rocmds_check_gfx1100/install
bash pack_from_lab.sh
docker build -t milvus-hip:v2.5.4-rocm context/
bash up.sh
bash smoke.sh
```

To publish a new lab pack: `bash push_ghcr.sh` (GitHub user `konkolchin`, PAT `write:packages`).

## From-source builder (hours, not this tag)

```bash
bash build_rocm_image.sh source
```

`APPLY_PACKER=1` and `OFFLOAD_ARCH=gfx1100` are the defaults. Not for GitHub-hosted CI.
The published `v2.5.4-rocm` tag is the **pack** path, not this builder.
