#!/usr/bin/env bash
# Tag the local HIP image and push to GHCR. Run on the AMD box that built it.
#
#   bash deployments/docker/standalone-rocm/push_ghcr.sh
#   GHCR_USER=krainov bash .../push_ghcr.sh
set -euo pipefail

LOCAL="${MILVUS_HIP_IMAGE:-milvus-hip:v2.5.4-rocm}"
USER_NAME="${GHCR_USER:-krainov}"
REMOTE="${MILVUS_HIP_GHCR:-ghcr.io/${USER_NAME}/milvus-hip:v2.5.4-rocm}"

if ! docker image inspect "${LOCAL}" >/dev/null 2>&1; then
  echo "ERROR: local image not found: ${LOCAL}" >&2
  echo "  Build first: bash pack_from_lab.sh && docker build -t ${LOCAL} context/" >&2
  exit 1
fi

echo "==> tag ${LOCAL} -> ${REMOTE}"
docker tag "${LOCAL}" "${REMOTE}"

if command -v gh >/dev/null 2>&1 && gh auth token >/dev/null 2>&1; then
  echo "==> docker login ghcr.io as ${USER_NAME} (gh auth token)"
  gh auth token | docker login ghcr.io -u "${USER_NAME}" --password-stdin
else
  echo "NOTE: gh not logged in. docker login ghcr.io yourself, then:" >&2
  echo "  docker push ${REMOTE}" >&2
fi

echo "==> docker push ${REMOTE}"
docker push "${REMOTE}"
echo "    pulled as: docker pull ${REMOTE}"
