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

mkdir -p "${CTX}/bin" "${CTX}/configs"
# Drop stale broken links (libhipblas.so.3 -> libhipblas.so.3.0.70002 with no target).
rm -rf "${CTX}/lib"
mkdir -p "${CTX}/lib"

# Copy the real ELF and the SONAME the binary asks for (libhipblas.so.3).
# Do not use [ -e ] on the source: that is false for a dangling symlink.
_copy_so() {
  local src="$1"
  local base dest real realbase
  [ -L "${src}" ] || [ -f "${src}" ] || return 0
  base="$(basename "${src}")"
  dest="${CTX}/lib/${base}"
  real="$(readlink -f "${src}" 2>/dev/null || true)"
  if [ -z "${real}" ] || [ ! -f "${real}" ]; then
    echo "WARNING: cannot resolve ${src}" >&2
    return 0
  fi
  realbase="$(basename "${real}")"
  cp -f "${real}" "${CTX}/lib/${realbase}"
  if [ "${realbase}" != "${base}" ]; then
    cp -f "${real}" "${dest}"
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

# hipVS / hipRAFT / Knowhere, plus the host ROCm math libs the July binary
# was linked against (hipblas, hipsolver, rocblas, ...). Copy the tree, not
# one SONAME at a time.
for _libdir in \
  "${HIPVS_PREFIX}/lib" \
  "${HIPVS_PREFIX}/lib64" \
  "${MILVUS_DIR}/internal/core/output/lib" \
  "${MILVUS_DIR}/lib" \
  "${MILVUS_DIR}/internal/core/output/lib64" \
  /opt/rocm/lib \
  /opt/rocm/lib64 \
  /opt/rocm/hip/lib \
  /opt/rocm/lib/hipblas \
  /opt/rocm/lib/hipsolver
do
  if [ -d "${_libdir}" ]; then
    echo "    libs from ${_libdir}"
    find "${_libdir}" -maxdepth 2 \( \
        -name '*.so' -o -name '*.so.*' \
      \) -print0 2>/dev/null \
      | while IFS= read -r -d '' _f; do _copy_so "${_f}"; done
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
    if [ -e "${_so}" ]; then
      _copy_so "${_so}"
    fi
  done < <(ldd "${target}" 2>/dev/null | awk '/=> \// {print $3} /^\// {print $1}')
}

_copy_ldd "${CTX}/bin/milvus"

_resolve_missing() {
  local name src
  name="$1"
  src="$(ldconfig -p 2>/dev/null | awk -v n="${name}" '$1 == n {print $NF; exit}')"
  if [ -z "${src}" ] || [ ! -e "${src}" ]; then
    src="$(find /opt/rocm /opt/rocm-* "${HIPVS_PREFIX}" "${WORKDIR}/install" \
      /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu \
      -name "${name}" -o -name "${name}.*" 2>/dev/null | head -1)"
  fi
  if [ -n "${src}" ]; then
    echo "    resolve ${name} <- ${src}"
    _copy_so "${src}"
  fi
}

for _pass in 1 2 3 4 5 6; do
  find "${CTX}/lib" -maxdepth 1 \( -name '*.so' -o -name '*.so.*' \) -print0 2>/dev/null \
    | while IFS= read -r -d '' _lib; do
        _copy_ldd "${_lib}"
      done
  _missing="$(LD_LIBRARY_PATH="${CTX}/lib" ldd "${CTX}/bin/milvus" 2>/dev/null | awk '/not found/ {print $1}')"
  if [ -z "${_missing}" ]; then
    echo "==> ldd clean (pass ${_pass})"
    break
  fi
  echo "==> still missing (pass ${_pass}): ${_missing}"
  for _n in ${_missing}; do
    _resolve_missing "${_n}"
  done
done

_missing="$(LD_LIBRARY_PATH="${CTX}/lib" ldd "${CTX}/bin/milvus" 2>/dev/null | grep 'not found' || true)"
if [ -n "${_missing}" ]; then
  echo "ERROR: milvus still has unresolved libs after pack:" >&2
  echo "${_missing}" >&2
  echo "  Host check: ldd ${CTX}/bin/milvus | grep not" >&2
  exit 1
fi

cp -a "${SCRIPT_DIR}/docker-entrypoint.sh" "${CTX}/docker-entrypoint.sh"
chmod 755 "${CTX}/docker-entrypoint.sh"

cp -a "${SCRIPT_DIR}/Dockerfile" "${CTX}/Dockerfile"
_nlink="$(find "${CTX}/lib" -maxdepth 1 -type l | wc -l)"
if [ "${_nlink}" != "0" ]; then
  echo "WARNING: ${_nlink} symlinks still in context/lib (Docker will not follow host /opt/rocm):" >&2
  find "${CTX}/lib" -maxdepth 1 -type l >&2
fi
echo "==> packed $(find "${CTX}/lib" -type f | wc -l) library files"
if [ "${DOCKER_BUILD:-0}" = "1" ]; then
  echo "==> docker build ${IMAGE}"
  docker build -t "${IMAGE}" "${CTX}"
  echo "    tagged ${IMAGE}"
else
  echo "Next (from this directory):"
  echo "  docker build -t ${IMAGE} context/"
  echo "  bash up.sh"
  echo "  bash smoke.sh"
fi
