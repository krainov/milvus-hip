#!/usr/bin/env bash
# Wipe the packed context and HIP container, pack every host ROCm/hipVS lib,
# rebuild the image, start standalone. etcd/minio stay up.
#
#   export MILVUS_DIR=$HOME/rocmds_check_gfx1100/milvus
#   export HIPVS_PREFIX=$HOME/rocmds_check_gfx1100/install
#   bash reset_from_scratch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${MILVUS_HIP_IMAGE:-milvus-hip:v2.5.4-rocm}"
export MILVUS_DIR="${MILVUS_DIR:-${HOME}/rocmds_check_gfx1100/milvus}"
export HIPVS_PREFIX="${HIPVS_PREFIX:-${HOME}/rocmds_check_gfx1100/install}"
export WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"

cd "${SCRIPT_DIR}"

echo "==> stop HIP standalone (keep etcd/minio)"
docker rm -f milvus-hip-standalone >/dev/null 2>&1 || true

echo "==> wipe old pack context"
rm -rf "${SCRIPT_DIR}/context"

echo "==> pack from ${MILVUS_DIR} + ${HIPVS_PREFIX} + /opt/rocm"
bash "${SCRIPT_DIR}/pack_from_lab.sh"

echo "==> packed glog/gflags (need gflags::FlagRegisterer, not google::)"
ls -l "${SCRIPT_DIR}/context/lib"/libglog.so* "${SCRIPT_DIR}/context/lib"/libgflags* 2>/dev/null || true
nm -D "${SCRIPT_DIR}/context/lib/libglog.so.1" 2>/dev/null | grep FlagRegisterer || true
nm -D "${SCRIPT_DIR}/context/lib/libgflags_gflagsns.so" 2>/dev/null | grep FlagRegisterer | head -5 || true

echo "==> host ldd with packed libs (must be empty)"
LD_LIBRARY_PATH="${SCRIPT_DIR}/context/lib" ldd "${SCRIPT_DIR}/context/bin/milvus" | grep 'not found' && {
  echo "ERROR: pack left unresolved libs; not building" >&2
  exit 1
} || echo "    ldd clean"

echo "==> rebuild image (no cache)"
docker rmi "${IMAGE}" >/dev/null 2>&1 || true
docker build --no-cache -t "${IMAGE}" "${SCRIPT_DIR}/context"

bash "${SCRIPT_DIR}/up.sh"
echo "==> waiting for healthz (standalone needs ~15-60s after DataNode register)"
ok=0
for i in $(seq 1 45); do
  if curl -sf http://127.0.0.1:9091/healthz >/dev/null 2>&1; then
    echo "    healthz OK (${i})"
    ok=1
    break
  fi
  sleep 2
done
echo "==> logs:"
docker logs milvus-hip-standalone 2>&1 | tail -20
if [ "${ok}" = "1" ]; then
  echo "Next: bash smoke.sh"
else
  echo "healthz not ready yet; wait and: bash smoke.sh"
fi
