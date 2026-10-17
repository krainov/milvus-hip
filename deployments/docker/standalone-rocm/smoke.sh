#!/usr/bin/env bash
# Lab gate before any public image tag.
# Run on the AMD host after: docker compose -f docker-compose.yml up -d
#
#   bash deployments/docker/standalone-rocm/smoke.sh
#   HARNESS_DIR=~/ann-harness-amd bash .../smoke.sh
set -euo pipefail

URI="${MILVUS_URI:-http://127.0.0.1:19530}"
HEALTHZ="${HEALTHZ_URL:-http://127.0.0.1:9091/healthz}"
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
  SKIP_START=1 SKIP_DEPS=1 bash "${HARNESS_DIR}/scripts/run_milvus_gpu_smoke.sh"
else
  echo "NOTE: harness not at ${HARNESS_DIR}; healthz passed."
  echo "      Clone konkolchin/ann-harness-amd and re-run for GPU_IVF_FLAT recall."
  echo "      Do not push a public tag until that smoke is green."
fi
