#!/usr/bin/env bash
# Lab gate for the published gfx1100 GPU_IVF_FLAT image.
# Run on the AMD host after: bash up.sh
#
#   bash deployments/docker/standalone-rocm/smoke.sh
#   HARNESS_DIR=~/ann-harness-amd bash .../smoke.sh
set -euo pipefail

URI="${MILVUS_URI:-http://127.0.0.1:19530}"
HEALTHZ="${HEALTHZ_URL:-http://127.0.0.1:9091/healthz}"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
HARNESS_DIR="${HARNESS_DIR:-}"
if [ -z "${HARNESS_DIR}" ]; then
  for d in \
    "${HOME}/ann-harness-amd" \
    "${HOME}/ann_harness_amd" \
    "${HOME}/ann-harness" \
    "${WORKDIR}/ann-harness-amd"
  do
    if [ -x "${d}/scripts/run_milvus_gpu_smoke.sh" ]; then
      HARNESS_DIR="${d}"
      break
    fi
  done
fi
HARNESS_DIR="${HARNESS_DIR:-${HOME}/ann-harness-amd}"

echo "==> waiting for ${HEALTHZ}"
ok=0
for i in $(seq 1 90); do
  if curl -sf "${HEALTHZ}" >/dev/null 2>&1; then
    echo "==> healthz ok (${i})"
    ok=1
    break
  fi
  sleep 2
done
if [ "${ok}" != "1" ]; then
  echo "ERROR: standalone not healthy. Is compose up? Is /dev/kfd visible?" >&2
  docker ps --format 'table {{.Names}}\t{{.Status}}' >&2 || true
  docker logs milvus-hip-standalone 2>&1 | tail -80 >&2 || true
  exit 1
fi

if [ -x "${HARNESS_DIR}/scripts/run_milvus_gpu_smoke.sh" ]; then
  echo "==> harness GPU_IVF_FLAT smoke (SKIP_START=1, server already in Docker)"
  echo "    HARNESS_DIR=${HARNESS_DIR}"
  export WORKDIR MILVUS_DIR="${MILVUS_DIR:-${WORKDIR}/milvus}"
  export HIPVS_PREFIX="${HIPVS_PREFIX:-${WORKDIR}/install}"
  if [ -z "${DATA_PATH:-}" ]; then
    for p in \
      "${HARNESS_DIR}/data/sift-128-euclidean.hdf5" \
      "${WORKDIR}/data/sift-128-euclidean.hdf5" \
      "${HOME}/sift-128-euclidean.hdf5"
    do
      if [ -f "${p}" ]; then DATA_PATH="${p}"; break; fi
    done
  fi
  if [ -n "${DATA_PATH:-}" ]; then
    export DATA_PATH
    echo "    DATA_PATH=${DATA_PATH}"
  fi
  export SMOKE_COLLECTION="${SMOKE_COLLECTION:-sift_gpu_ivf_smoke_${RANDOM}}"
  echo "    SMOKE_COLLECTION=${SMOKE_COLLECTION}"
  SKIP_START=1 SKIP_DEPS=1 bash "${HARNESS_DIR}/scripts/run_milvus_gpu_smoke.sh"
else
  echo "NOTE: harness not at ${HARNESS_DIR}; healthz passed. Server is up on :19530."
  echo "      GPU_IVF_FLAT still needs the harness + SIFT hdf5:"
  echo "        git clone https://github.com/konkolchin/ann-harness-amd.git ~/ann-harness-amd"
  echo "        mkdir -p ~/ann-harness-amd/data"
  echo "        wget --no-proxy https://ann-benchmarks.com/sift-128-euclidean.hdf5 -O ~/ann-harness-amd/data/sift-128-euclidean.hdf5"
  echo "        bash $0"
  echo "      Do not push a public tag until that smoke is green."
fi
