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

# Copy shared libs the binary actually needs. The July host ROCm SONAMEs
# (e.g. libhipblas.so.3) are often missing from rocm/dev-ubuntu Docker tags,
# so do not skip /opt/rocm — only skip glibc/libstdc++ from the Ubuntu image.
_skip_so() {
  case "$1" in
    */ld-linux*|*/linux-vdso*|*/libc.so*|*/libm.so*|*/libpthread.so*|*/libdl.so*|*/librt.so*|*/libresolv.so*|*/libgcc_s.so*|*/libstdc++.so*)
      return 0 ;;
  esac
  return 1
}

_copy_ldd() {
  local target="$1"
  command -v ldd >/dev/null || return 0
  local _so
  while IFS= read -r _so; do
    [ -z "${_so}" ] && continue
    _skip_so "${_so}" && continue
    if [ -f "${_so}" ]; then
      cp -a "${_so}" "${CTX}/lib/" 2>/dev/null || true
    fi
  done < <(ldd "${target}" 2>/dev/null | awk '/=> \// {print $3} /^\// {print $1}')
}

_copy_ldd "${CTX}/bin/milvus"
# hipblas pulls rocblas etc.; walk libs already in the context a couple of times.
for _pass in 1 2 3; do
  find "${CTX}/lib" -maxdepth 1 \( -name '*.so' -o -name '*.so.*' \) -print0 2>/dev/null \
    | while IFS= read -r -d '' _lib; do
        _copy_ldd "${_lib}"
      done
done

if command -v ldd >/dev/null; then
  _missing="$(ldd "${CTX}/bin/milvus" 2>/dev/null | grep 'not found' || true)"
  # Re-check with packed lib dir first, like the container LD_LIBRARY_PATH.
  _missing="$(LD_LIBRARY_PATH="${CTX}/lib:${LD_LIBRARY_PATH:-}" ldd "${CTX}/bin/milvus" 2>/dev/null | grep 'not found' || true)"
  if [ -n "${_missing}" ]; then
    echo "WARNING: still unresolved after pack:" >&2
    echo "${_missing}" >&2
  fi
fi

cp -a "${SCRIPT_DIR}/docker-entrypoint.sh" "${CTX}/docker-entrypoint.sh"
chmod 755 "${CTX}/docker-entrypoint.sh"

cp -a "${SCRIPT_DIR}/Dockerfile" "${CTX}/Dockerfile"
echo "==> packed $(find "${CTX}/lib" -type f | wc -l) library files"
if [ "${DOCKER_BUILD:-0}" = "1" ]; then
  echo "==> docker build ${IMAGE}"
  docker build -t "${IMAGE}" "${CTX}"
  echo "    tagged ${IMAGE}"
else
  echo "Next (from this directory):"
  echo "  docker build -t ${IMAGE} context/"
  echo "  docker-compose -f docker-compose.yml up -d"
  echo "  # or: docker compose -f docker-compose.yml up -d"
fi
