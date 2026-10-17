# Milvus HIP standalone (ROCm Docker)

This is the AMD analogue of official Milvus GPU Docker. Official
`milvusdb/milvus` GPU tags are **CUDA**. They do not give GPU search on a Radeon.

Stock [pymilvus](https://github.com/milvus-io/pymilvus) talks to port **19530**.
There is no HIP build of pymilvus.

Validated path: RX 7900 XTX (`gfx1100`), ROCm 7.0.2. Instinct is untested.
Do not apply the harness IVF packer patch on wave64 Instinct GPUs.

## Phase 1 — pack a lab build (first green image)

On the AMD box, after Layer-3 (`ann-harness-amd` `build_milvus_layer3.sh`):

```bash
# Stop leftover official standalone (wrong libcuvs mix-up)
docker rm -f milvus-standalone 2>/dev/null || true

cd deployments/docker/standalone-rocm
bash pack_from_lab.sh          # or: bash build_rocm_image.sh pack
docker compose up -d
bash smoke.sh
```

Do not push `ghcr.io/krainov/milvus-hip:v2.5.4-rocm` until `smoke.sh` is green.

## Phase 2 — compile inside Docker (hours)

```bash
bash build_rocm_image.sh source
```

`APPLY_PACKER=1` and `OFFLOAD_ARCH=gfx1100` are the defaults. Not for GitHub-hosted CI.

## Client

```bash
pip install pymilvus
```

```python
from pymilvus import connections
connections.connect(host="127.0.0.1", port="19530")
```
