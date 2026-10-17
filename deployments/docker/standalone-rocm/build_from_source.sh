#!/usr/bin/env bash
# From-source HIP stack inside Dockerfile.build. Hours. Not for GitHub-hosted CI.
# Default: gfx1100 + IVF packer (wave32). Instinct: APPLY_PACKER=0 OFFLOAD_ARCH=gfx90a
# and do not apply the harness packer patches.
set -euo pipefail

export WORKDIR="${WORKDIR:-/opt/hip-build}"
export INSTALL_PREFIX="${INSTALL_PREFIX:-${WORKDIR}/install}"
export HIPVS_PREFIX="${HIPVS_PREFIX:-${INSTALL_PREFIX}}"
export MILVUS_DIR="${MILVUS_DIR:-${WORKDIR}/milvus}"
export KNOWHERE_DIR="${KNOWHERE_DIR:-${WORKDIR}/knowhere}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export ARCH="${OFFLOAD_ARCH:-gfx1100}"
export BRANCH="${HIPVS_BRANCH:-release/rocmds-25.10}"
APPLY_PACKER="${APPLY_PACKER:-1}"
HARNESS_DIR="${WORKDIR}/ann-harness-amd"

HARNESS_REPO="${HARNESS_REPO:-https://github.com/konkolchin/ann-harness-amd.git}"
MILVUS_REPO="${MILVUS_REPO:-https://github.com/krainov/milvus-hip.git}"
MILVUS_TAG="${MILVUS_TAG:-v2.5.4}"
KNOWHERE_REPO="${KNOWHERE_REPO:-https://github.com/konkolchin/knowhere-hip.git}"
KNOWHERE_TAG="${KNOWHERE_TAG:-2.5}"

mkdir -p "${WORKDIR}" "${INSTALL_PREFIX}" "${WORKDIR}/logs"
cd "${WORKDIR}"

if [ ! -d "${HARNESS_DIR}/.git" ]; then
  git clone --depth 1 "${HARNESS_REPO}" "${HARNESS_DIR}"
fi
if [ ! -d "${MILVUS_DIR}/.git" ]; then
  git clone -b "${MILVUS_TAG}" --depth 1 "${MILVUS_REPO}" "${MILVUS_DIR}"
fi
if [ ! -d "${KNOWHERE_DIR}/.git" ]; then
  git clone -b "${KNOWHERE_TAG}" --depth 1 "${KNOWHERE_REPO}" "${KNOWHERE_DIR}"
fi

echo "==> Layer 1 hipRAFT / hipVS (${ARCH}, ${BRANCH})"
bash "${HARNESS_DIR}/scripts/check_rocmds_gfx1100.sh"
# The Layer-1 script always passes -DUSE_WARPSIZE_32=ON (gfx1100). Do not use this
# builder for Instinct without changing that flag; hipVS already targets wave64.
if [ ! -f "${INSTALL_PREFIX}/lib/cmake/cuvs/cuvs-config.cmake" ] && \
   [ ! -f "${INSTALL_PREFIX}/lib64/cmake/cuvs/cuvs-config.cmake" ]; then
  echo "ERROR: hipVS did not install cuvs-config.cmake under ${INSTALL_PREFIX}" >&2
  exit 1
fi

if [ "${APPLY_PACKER}" = "1" ]; then
  echo "==> IVF packer warp32 patches (gfx1100 only)"
  bash "${HARNESS_DIR}/scripts/apply_hipvs_layer15_patches.sh" "${WORKDIR}/hipVS"
else
  echo "==> APPLY_PACKER=0: not applying harness IVF packer patches"
fi

echo "==> Layer 2 Knowhere"
SKIP_GIT_PULL=1 bash "${HARNESS_DIR}/scripts/build_knowhere_layer2.sh"

echo "==> Layer 3 Milvus HIP"
SKIP_CLONE=1 bash "${HARNESS_DIR}/scripts/build_milvus_layer3.sh"

echo "==> pack /out for the runtime stage"
export CTX=/out
export DOCKER_BUILD=0
mkdir -p /out
# Reuse pack logic against WORKDIR paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTX=/out bash "${SCRIPT_DIR}/pack_from_lab.sh"
echo "BUILD_FROM_SOURCE_OK"
