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

echo "==> docker run ${NAME} (${IMAGE})"
docker run -d --name "${NAME}" \
  --network "${NET}" \
  --security-opt seccomp:unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -e MINIO_REGION=us-east-1 \
  -e ETCD_ENDPOINTS=etcd:2379 \
  -e MINIO_ADDRESS=minio:9000 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e ROCR_VISIBLE_DEVICES=0 \
  -p 19530:19530 \
  -p 9091:9091 \
  -v "${VOL}:/var/lib/milvus" \
  "${IMAGE}" \
  milvus run standalone

echo "    docker logs -f ${NAME}"
echo "    curl -sf http://127.0.0.1:9091/healthz && echo OK"
echo "    bash ${SCRIPT_DIR}/smoke.sh"
