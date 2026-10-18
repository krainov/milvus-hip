#!/usr/bin/env bash
# Start HIP standalone without apt docker-compose inspecting the image.
# Ubuntu python-compose 1.x dies with KeyError: ContainerConfig on this image.
# etcd/minio can stay on docker-compose; milvus is `docker run`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${MILVUS_HIP_IMAGE:-milvus-hip:v2.5.4-rocm}"
NAME="${MILVUS_HIP_NAME:-milvus-hip-standalone}"
NET="${MILVUS_HIP_NET:-milvus-hip}"
VOL="${DOCKER_VOLUME_DIRECTORY:-${SCRIPT_DIR}}/volumes/milvus"

cd "${SCRIPT_DIR}"
mkdir -p "${VOL}"

if docker-compose -f docker-compose.yml ps etcd 2>/dev/null | grep -q milvus-etcd; then
  echo "==> etcd/minio already up"
else
  echo "==> docker-compose up etcd minio"
  docker-compose -f docker-compose.yml up -d etcd minio
fi

docker rm -f "${NAME}" >/dev/null 2>&1 || true

# Recreate standalone gets a new proxy node ID. etcd still has the old
# session → drop_collection fails with node not match expected=2 actual=3.
if docker-compose -f docker-compose.yml ps etcd 2>/dev/null | grep -q milvus-etcd; then
  echo "==> restart etcd (drop stale proxy session)"
  docker-compose -f docker-compose.yml restart etcd
  sleep 5
fi

# AMDGPU_IDS: host file the ROCm image often omits (startup: No such file).
_ids=""
for p in /opt/amdgpu/share/libdrm/amdgpu.ids /usr/share/libdrm/amdgpu.ids; do
  if [ -f "${p}" ]; then _ids="${p}"; break; fi
done

echo "==> docker run ${NAME} (${IMAGE})"
# --ipc=host: hipVS IVF search uses GPU IPC; index build can succeed without it.
_run=(docker run -d --name "${NAME}"
  --network "${NET}"
  --ipc=host
  --cap-add SYS_PTRACE
  --security-opt seccomp:unconfined
  --device /dev/kfd
  --device /dev/dri
  --group-add video
  -e MINIO_REGION=us-east-1
  -e ETCD_ENDPOINTS=etcd:2379
  -e MINIO_ADDRESS=minio:9000
  -e HIP_VISIBLE_DEVICES=0
  -e ROCR_VISIBLE_DEVICES=0
  -p 19530:19530
  -p 9091:9091
  -v "${VOL}:/var/lib/milvus"
)
if [ -n "${_ids}" ]; then
  echo "    mount amdgpu.ids from ${_ids}"
  _run+=(-v "${_ids}:/opt/amdgpu/share/libdrm/amdgpu.ids:ro")
fi
"${_run[@]}" "${IMAGE}" milvus run standalone

echo "    docker logs -f ${NAME}"
echo "    curl -sf http://127.0.0.1:9091/healthz && echo OK"
echo "    bash ${SCRIPT_DIR}/smoke.sh"
