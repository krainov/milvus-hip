#!/usr/bin/env bash
# Assemble a Docker build context from a known-good Layer-3 HIP Milvus prefix.
# Run on the AMD lab box after:
#   bash scripts/build_milvus_layer3.sh   # from ann-harness-amd
#
# Usage:
#   bash deployments/docker/standalone-rocm/pack_from_lab.sh
#   DOCKER_BUILD=1 bash deployments/docker/standalone-rocm/pack_from_lab.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${WORKDIR}/install}"
HIPVS_PREFIX="${HIPVS_PREFIX:-${INSTALL_PREFIX}}"
MILVUS_DIR="${MILVUS_DIR:-${WORKDIR}/milvus}"
CTX="${CTX:-${SCRIPT_DIR}/context}"
IMAGE="${MILVUS_HIP_IMAGE:-milvus-hip:v2.5.4-rocm}"

mkdir -p "${CTX}/bin" "${CTX}/configs" "${CTX}/lib"

copy_file() {
  local src="$1" dest="$2"
  if [ -e "${src}" ]; then
    cp -a "${src}" "${dest}"
  fi
}

echo "==> packing into ${CTX}"

_bin=""
for c in \
  "${MILVUS_DIR}/bin/milvus" \
  "${MILVUS_DIR}/internal/core/output/bin/milvus" \
  "${MILVUS_DIR}/cmake_build/milvus" \
  "${MILVUS_DIR}/out/bin/milvus"
do
  if [ -x "${c}" ]; then _bin="${c}"; break; fi
done
if [ -z "${_bin}" ]; then
  echo "ERROR: milvus binary not found under ${MILVUS_DIR}" >&2
  echo "  Build first: bash <harness>/scripts/build_milvus_layer3.sh" >&2
  exit 1
fi
if [ ! -d "${HIPVS_PREFIX}/lib" ] && [ ! -d "${HIPVS_PREFIX}/lib64" ]; then
  echo "ERROR: HIPVS_PREFIX has no lib/: ${HIPVS_PREFIX}" >&2
  exit 1
fi
cp -a "${_bin}" "${CTX}/bin/milvus"
chmod 755 "${CTX}/bin/milvus"
echo "    milvus: ${_bin}"

if [ -d "${MILVUS_DIR}/configs" ]; then
  cp -a "${MILVUS_DIR}/configs/." "${CTX}/configs/"
elif [ -d "${SCRIPT_DIR}/../../../configs" ]; then
  cp -a "${SCRIPT_DIR}/../../../configs/." "${CTX}/configs/"
else
  echo "ERROR: no configs/ next to milvus tree" >&2
  exit 1
fi

# hipVS / hipRAFT / Knowhere shared libs from the HIP prefix.
for _libdir in \
  "${HIPVS_PREFIX}/lib" \
  "${HIPVS_PREFIX}/lib64" \
  "${MILVUS_DIR}/internal/core/output/lib" \
  "${MILVUS_DIR}/lib" \
  "${MILVUS_DIR}/internal/core/output/lib64"
do
  if [ -d "${_libdir}" ]; then
    echo "    libs from ${_libdir}"
    find "${_libdir}" -maxdepth 1 \( -name '*.so' -o -name '*.so.*' \) -exec cp -a {} "${CTX}/lib/" \;
  fi
done

# Transitive NEEDED entries of the milvus binary (skip glibc / ROCm already in the base image).
if command -v ldd >/dev/null; then
  while IFS= read -r _so; do
    [ -z "${_so}" ] && continue
    case "${_so}" in
      /lib/*|/lib64/*|/usr/lib/*|/usr/lib64/*|/opt/rocm/*) continue ;;
    esac
    if [ -f "${_so}" ]; then
      cp -a "${_so}" "${CTX}/lib/" || true
    fi
  done < <(ldd "${CTX}/bin/milvus" 2>/dev/null | awk '/=> \// {print $3} /^\// {print $1}')
fi

cp -a "${SCRIPT_DIR}/docker-entrypoint.sh" "${CTX}/docker-entrypoint.sh"
chmod 755 "${CTX}/docker-entrypoint.sh"

echo "==> packed $(find "${CTX}/lib" -type f | wc -l) library files"
if [ "${DOCKER_BUILD:-0}" = "1" ]; then
  echo "==> docker build ${IMAGE}"
  docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE}" "${CTX}"
  echo "    tagged ${IMAGE}"
else
  echo "Next:"
  echo "  docker build -f ${SCRIPT_DIR}/Dockerfile -t ${IMAGE} ${CTX}"
  echo "  docker compose -f ${SCRIPT_DIR}/docker-compose.yml up -d"
fi
