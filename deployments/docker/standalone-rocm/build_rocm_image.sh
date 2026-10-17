#!/usr/bin/env bash
# Build the Milvus HIP runtime image on the AMD lab box.
#
#   bash build_rocm_image.sh pack      # Phase 1: pack Layer-3 prefix (default)
#   bash build_rocm_image.sh source    # Phase 2: compile hipVS/Knowhere/Milvus in Docker
#
# GitHub-hosted runners cannot finish "source". Do not push a public tag until
# smoke.sh passes on a Radeon with /dev/kfd.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-pack}"
IMAGE="${MILVUS_HIP_IMAGE:-milvus-hip:v2.5.4-rocm}"

case "${MODE}" in
  pack)
    DOCKER_BUILD=1 bash "${SCRIPT_DIR}/pack_from_lab.sh"
    ;;
  source)
    echo "==> docker build -f Dockerfile.build (hours)"
    docker build \
      -f "${SCRIPT_DIR}/Dockerfile.build" \
      -t "${IMAGE}" \
      --build-arg APPLY_PACKER="${APPLY_PACKER:-1}" \
      --build-arg OFFLOAD_ARCH="${OFFLOAD_ARCH:-gfx1100}" \
      "${SCRIPT_DIR}"
    ;;
  *)
    echo "Usage: $0 [pack|source]" >&2
    exit 2
    ;;
esac

echo "Image: ${IMAGE}"
echo "Next: docker compose -f ${SCRIPT_DIR}/docker-compose.yml up -d"
echo "      bash ${SCRIPT_DIR}/smoke.sh"
