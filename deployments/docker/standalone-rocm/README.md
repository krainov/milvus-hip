# Milvus HIP standalone (ROCm Docker)

Tech preview image for Milvus **v2.5.4** GPU ANN on AMD **HIP/ROCm**.

Official `milvusdb/milvus` GPU tags are **CUDA**. They do not give GPU search
on a Radeon. Stock [pymilvus](https://github.com/milvus-io/pymilvus) talks to
port **19530**. There is no HIP build of pymilvus.

## What this image claims

Image: `ghcr.io/konkolchin/milvus-hip:v2.5.4-rocm`

| Claim | Status |
|-------|--------|
| Milvus | v2.5.4 HIP (prebuilt binary packed into this image) |
| GPU | RX 7900 XTX (`gfx1100`), ROCm 7.0.2 |
| `GPU_IVF_FLAT` | **Verified** — sealed `GPU_CUVS_IVF_FLAT` build/load/search; slice-exact recall@10 = 1.0 at nprobe 128 (50k×128 SIFT) |
| `GPU_IVF_PQ` | **Not claimed** for this image |
| `GPU_CAGRA` | **Not claimed** |
| Instinct | **Not claimed** (untested). Do not apply the harness wave32 IVF packer on wave64 Instinct |

## Pull and run

You need an AMD GPU with ROCm, Docker, and `/dev/kfd` (typical on a Radeon
workstation). Clone this repo only for `up.sh` / compose; the server binary
is in the image.

```bash
git clone -b v2.5.4 https://github.com/krainov/milvus-hip.git
cd milvus-hip/deployments/docker/standalone-rocm

docker pull ghcr.io/konkolchin/milvus-hip:v2.5.4-rocm
docker rm -f milvus-standalone milvus-hip-standalone 2>/dev/null || true

export MILVUS_HIP_IMAGE=ghcr.io/konkolchin/milvus-hip:v2.5.4-rocm
bash up.sh
bash smoke.sh
```

`up.sh` starts etcd and MinIO, then `docker run` with `--device /dev/kfd` and
`--ipc=host`. Ubuntu apt `docker-compose` 1.x hits `KeyError: ContainerConfig`
on this image — use `up.sh`.

```python
from pymilvus import connections
connections.connect(host="127.0.0.1", port="19530")
```

## Rebuild from a host Milvus tree (optional)

Most users should **pull the image above**. Rebuild only if you compiled HIP
Milvus yourself.

1. Clone the harness: [konkolchin/ann-harness-amd](https://github.com/konkolchin/ann-harness-amd).
2. Build Milvus on the AMD host with
   [`scripts/build_milvus_layer3.sh`](https://github.com/konkolchin/ann-harness-amd/blob/master/scripts/build_milvus_layer3.sh)
   (that script is the host compile; it is not required to *run* the published image).
3. Pack those binaries into a new Docker image:

```bash
docker rm -f milvus-standalone 2>/dev/null || true
cd milvus-hip/deployments/docker/standalone-rocm
export MILVUS_DIR=$HOME/rocmds_check_gfx1100/milvus
export HIPVS_PREFIX=$HOME/rocmds_check_gfx1100/install
bash pack_from_lab.sh
docker build -t milvus-hip:local context/
export MILVUS_HIP_IMAGE=milvus-hip:local
bash up.sh
bash smoke.sh
```

That local tag is enough to run. You do not need GitHub Container Registry.
To keep the image on another machine:

```bash
docker save milvus-hip:local | gzip > milvus-hip-local.tar.gz
# on the other host:
gunzip -c milvus-hip-local.tar.gz | docker load
export MILVUS_HIP_IMAGE=milvus-hip:local
bash up.sh
```

Optional: `docker tag milvus-hip:local ghcr.io/<your-github-user>/milvus-hip:v2.5.4-rocm` and
`docker push` to **your** GHCR namespace (PAT with `write:packages`). The published
`ghcr.io/konkolchin/milvus-hip:v2.5.4-rocm` tag is only for this project's
maintainers.

## From-source builder (hours, not this tag)

```bash
bash build_rocm_image.sh source
```

`APPLY_PACKER=1` and `OFFLOAD_ARCH=gfx1100` are the defaults. Not for GitHub-hosted CI.
The published `v2.5.4-rocm` tag is the **pack** path, not this builder.
