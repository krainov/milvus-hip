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
  # Conan glog wants gflags::FlagRegisterer. Apt/ROCm gflags export google::.
  # Never copy those during the bulk/ldd pass — pin step below picks the pair.
  case "$(basename "${src}")" in
    libglog.so*|libgflags.so*|libgflags_nothreads.so*|libgflags_gflagsns.so*)
      case "${src}" in
        /opt/rocm/*|/usr/*|/lib/*|/lib64/*) return 0 ;;
      esac
      ;;
  esac
  # ldd of files already in context/lib points back here — skip or cp errors
  # "are the same file".
  case "${src}" in
    "${CTX}/lib"/*|"${CTX}/lib/"*) return 0 ;;
  esac
  base="$(basename "${src}")"
  dest="${CTX}/lib/${base}"
  real="$(readlink -f "${src}" 2>/dev/null || true)"
  if [ -z "${real}" ] || [ ! -f "${real}" ]; then
    echo "WARNING: cannot resolve ${src}" >&2
    return 0
  fi
  case "${real}" in
    "${CTX}/lib"/*) return 0 ;;
  esac
  realbase="$(basename "${real}")"
  if [ ! "${real}" -ef "${CTX}/lib/${realbase}" ] 2>/dev/null; then
    cp -f "${real}" "${CTX}/lib/${realbase}"
  fi
  if [ "${realbase}" != "${base}" ] && [ ! "${real}" -ef "${dest}" ] 2>/dev/null; then
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

# hipVS / Knowhere / Milvus output (includes the Conan glog/gflags the binary
# was linked with). Do not glob all of /opt/rocm/lib — that copies LLVM/glog
# and then glog looks up the wrong gflags (FlagRegisterer).
for _libdir in \
  "${HIPVS_PREFIX}/lib" \
  "${HIPVS_PREFIX}/lib64" \
  "${MILVUS_DIR}/internal/core/output/lib" \
  "${MILVUS_DIR}/lib" \
  "${MILVUS_DIR}/internal/core/output/lib64"
do
  if [ -d "${_libdir}" ]; then
    echo "    libs from ${_libdir}"
    find "${_libdir}" -maxdepth 1 \( -name '*.so' -o -name '*.so.*' \) -print0 2>/dev/null \
      | while IFS= read -r -d '' _f; do _copy_so "${_f}"; done
  fi
done

# Host ROCm math/runtime only (hipblas, hipsolver, rocblas, HIP runtime).
for _libdir in /opt/rocm/lib /opt/rocm/lib64 /opt/rocm/hip/lib; do
  if [ -d "${_libdir}" ]; then
    echo "    ROCm hip/roc from ${_libdir}"
    find "${_libdir}" -maxdepth 2 \( \
        -name 'libhip*.so*' -o -name 'libroc*.so*' \
        -o -name 'libamdhip*.so*' -o -name 'libhsa*.so*' \
        -o -name 'libhsakmt*.so*' -o -name 'librocm*.so*' \
        -o -name 'libamd_comgr*.so*' \
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
  case "${name}" in
    libglog.so*|libgflags.so*|libgflags_nothreads.so*)
      echo "    skip apt/rocm resolve for ${name} (ABI pin later)"
      return 0
      ;;
  esac
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

echo "==> pin glog/gflags to gflags::FlagRegisterer (not apt google::)"
rm -f "${CTX}/lib"/libglog.so* "${CTX}/lib"/libgflags.so* \
  "${CTX}/lib"/libgflags_nothreads.so* "${CTX}/lib"/libgflags_gflagsns.so*

_gflags_ns() {
  nm -D "$1" 2>/dev/null | grep -q '_ZN6gflags14FlagRegisterer'
}

_install_gflags_elf() {
  local real="$1"
  local soname=""
  [ -f "${real}" ] || return 1
  _gflags_ns "${real}" || return 1
  echo "    gflags(gflags::) ${real}"
  _copy_so "${real}"
  cp -f "${real}" "${CTX}/lib/libgflags_gflagsns.so"
  soname="$(objdump -p "${real}" 2>/dev/null | awk '/SONAME/ {print $2; exit}')"
  if [ -z "${soname}" ]; then
    soname="$(readelf -d "${real}" 2>/dev/null | awk '/SONAME/ {gsub(/[\[\]]/,"",$NF); print $NF; exit}')"
  fi
  # glog DT_NEEDED is usually libgflags.so.2.2 even when the ELF basename differs.
  for _n in ${soname} libgflags.so.2.2 libgflags.so.2 libgflags.so "$(basename "${real}")"; do
    [ -n "${_n}" ] || continue
    cp -f "${real}" "${CTX}/lib/${_n}"
  done
  return 0
}

_glog="$(ldd "${_bin}" 2>/dev/null | awk '/libglog\.so/ && /=> \// {print $3; exit}')"
case "${_glog}" in
  /opt/rocm/*|/usr/*|/lib/*|/lib64/*) _glog="" ;;
esac
if [ -z "${_glog}" ]; then
  for _cand in \
    "${MILVUS_DIR}/internal/core/output/lib"/libglog.so.1 \
    "${MILVUS_DIR}/internal/core/output/lib"/libglog.so \
    "${MILVUS_DIR}/lib"/libglog.so.1 \
    "${HIPVS_PREFIX}/lib"/libglog.so.1 \
    "${HIPVS_PREFIX}/lib64"/libglog.so.1
  do
    [ -e "${_cand}" ] || continue
    _glog="$(readlink -f "${_cand}")"
    break
  done
fi
if [ -z "${_glog}" ]; then
  for _kh in "${MILVUS_DIR}/internal/core/output/lib"/libknowhere*.so*; do
    [ -e "${_kh}" ] || continue
    _glog="$(ldd "${_kh}" 2>/dev/null | awk '/libglog\.so/ && /=> \// {print $3; exit}')"
    case "${_glog}" in
      /opt/rocm/*|/usr/*|/lib/*|/lib64/*) _glog="" ;;
    esac
    [ -n "${_glog}" ] && break
  done
fi
if [ -n "${_glog}" ]; then
  echo "    glog ${_glog}"
  _copy_so "${_glog}"
fi

_gflags_ok=0
for _cand in \
  "${WORKDIR}/libshims/libgflags_gflagsns.so" \
  "${MILVUS_DIR}/internal/core/output/lib"/libgflags.so.2.2 \
  "${MILVUS_DIR}/internal/core/output/lib"/libgflags.so \
  "${MILVUS_DIR}/lib"/libgflags.so.2.2 \
  "${HIPVS_PREFIX}/lib"/libgflags.so.2.2 \
  "${HIPVS_PREFIX}/lib"/libgflags.so \
  "${HIPVS_PREFIX}/lib64"/libgflags.so.2.2
do
  [ -e "${_cand}" ] || continue
  if _install_gflags_elf "$(readlink -f "${_cand}")"; then
    _gflags_ok=1
    break
  fi
done
if [ "${_gflags_ok}" = "0" ]; then
  while IFS= read -r _cand; do
    [ -n "${_cand}" ] || continue
    if _install_gflags_elf "${_cand}"; then
      _gflags_ok=1
      break
    fi
  done < <(find "${HOME}/.conan" "${HOME}/.conan2" "${WORKDIR}" \
    \( -name 'libgflags.so*' -o -name 'libgflags_gflagsns.so' \) -type f 2>/dev/null | head -40)
fi

# July box has no leftover libshims file. Rebuild from Conan gflags .a
# (same recipe as the Knowhere GPU test runner).
if [ "${_gflags_ok}" = "0" ]; then
  mkdir -p "${WORKDIR}/libshims"
  set +e
  _gflags_a="$(ls -1 \
    "${HOME}"/.conan/data/gflags/*/_/_/package/*/lib/libgflags_nothreads.a \
    "${HOME}"/.conan/data/gflags/*/_/_/package/*/lib/libgflags.a \
    "${HOME}"/.conan2/p/*/p/lib/libgflags_nothreads.a \
    "${HOME}"/.conan2/p/*/p/lib/libgflags.a \
    2>/dev/null | head -1)"
  set -e
  if [ -n "${_gflags_a}" ] && [ -f "${_gflags_a}" ]; then
    echo "    build gflags:: shim from ${_gflags_a}"
    _tmp_so="${WORKDIR}/libshims/libgflags_shim_tmp.so"
    _shim="${WORKDIR}/libshims/libgflags_gflagsns.so"
    g++ -shared -fPIC -o "${_tmp_so}" \
      -Wl,--whole-archive "${_gflags_a}" -Wl,--no-whole-archive \
      -lpthread -lrt
    if nm -D "${_tmp_so}" 2>/dev/null | grep -q '_ZN6gflags14FlagRegisterer'; then
      mv -f "${_tmp_so}" "${_shim}"
    elif nm -D "${_tmp_so}" 2>/dev/null | grep -q '_ZN6google14FlagRegisterer'; then
      _map="${WORKDIR}/libshims/gflags_redefine.map"
      nm -D "${_tmp_so}" | awk '{print $NF}' | grep '6google' | sort -u | while read -r _sym; do
        printf '%s %s\n' "${_sym}" "${_sym//6google/6gflags}"
      done > "${_map}"
      echo "    rename google:: -> gflags:: ($(wc -l < "${_map}") symbols)"
      objcopy --redefine-syms="${_map}" "${_tmp_so}" "${_shim}"
      rm -f "${_tmp_so}" "${_map}"
    else
      echo "ERROR: FlagRegisterer not in Conan gflags after linking ${_tmp_so}" >&2
      nm -D "${_tmp_so}" 2>/dev/null | grep -i FlagRegisterer | head >&2 || true
      exit 1
    fi
    if _install_gflags_elf "${_shim}"; then
      _gflags_ok=1
    fi
  else
    echo "    no Conan libgflags*.a under ~/.conan"
  fi
fi

if [ -f "${CTX}/lib/libglog.so.1" ] && \
   nm -D "${CTX}/lib/libglog.so.1" 2>/dev/null | grep -q ' U .*_ZN6gflags14FlagRegisterer'
then
  if ! find "${CTX}/lib" -maxdepth 1 \( -name 'libgflags*' \) -print0 2>/dev/null \
      | xargs -0 -r nm -D 2>/dev/null | grep -q '_ZN6gflags14FlagRegisterer'
  then
    echo "ERROR: packed libglog.so.1 needs gflags::FlagRegisterer, but no packed" >&2
    echo "  libgflags exports it (apt/ROCm gflags use google:: and will not work)." >&2
    echo "  Expected Conan archive: ~/.conan/data/gflags/.../lib/libgflags*.a" >&2
    echo "  ls ~/.conan/data/gflags/*/_/_/package/*/lib/" >&2
    exit 1
  fi
fi
if [ "${_gflags_ok}" = "1" ]; then
  echo "    FlagRegisterer pin ok"
elif [ -f "${CTX}/lib/libglog.so.1" ]; then
  echo "WARNING: no gflags:: libgflags found; container will likely die on FlagRegisterer" >&2
fi

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
