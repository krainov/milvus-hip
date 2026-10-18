#!/usr/bin/env bash
# Tag the local HIP image and push to GHCR. Run on the AMD box that built it.
#
#   export GHCR_TOKEN=ghp_...   # PAT with write:packages, user krainov
#   bash deployments/docker/standalone-rocm/push_ghcr.sh
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

_logged_in=0
if [ -n "${GHCR_TOKEN:-}" ]; then
  echo "==> docker login ghcr.io as ${USER_NAME} (GHCR_TOKEN)"
  printf '%s\n' "${GHCR_TOKEN}" | docker login ghcr.io -u "${USER_NAME}" --password-stdin
  _logged_in=1
elif command -v gh >/dev/null 2>&1 && gh auth token >/dev/null 2>&1; then
  echo "==> docker login ghcr.io as ${USER_NAME} (gh auth token)"
  gh auth token | docker login ghcr.io -u "${USER_NAME}" --password-stdin
  _logged_in=1
fi

if [ "${_logged_in}" != "1" ]; then
  echo "ERROR: not logged in to ghcr.io. Create a GitHub PAT (classic) as ${USER_NAME}" >&2
  echo "  with scope write:packages (and read:packages)." >&2
  echo "  https://github.com/settings/tokens" >&2
  echo "Then:" >&2
  echo "  export GHCR_TOKEN=ghp_..." >&2
  echo "  bash $0" >&2
  echo "Or:" >&2
  echo "  echo \"\$GHCR_TOKEN\" | docker login ghcr.io -u ${USER_NAME} --password-stdin" >&2
  echo "  docker push ${REMOTE}" >&2
  exit 1
fi

echo "==> docker push ${REMOTE}"
docker push "${REMOTE}"
echo "    docker pull ${REMOTE}"
echo "    If the package is private: GitHub → Packages → milvus-hip → Public"
