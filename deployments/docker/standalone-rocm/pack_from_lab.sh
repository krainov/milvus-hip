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

# hipBLASLt / rocBLAS look for Tensile kernels next to the .so we packed
# into /milvus/lib. Without this, IVF_FLAT search (dim=128) dies with
# cublasLtMatmulAlgoGetHeuristic CUBLAS_STATUS_INVALID_VALUE; dim=8 can
# skip Lt and still work.
_copy_tensile() {
  local src="$1" dest="$2"
  if [ -d "${src}" ]; then
    echo "    tensile ${src} -> ${CTX}/lib/${dest}"
    mkdir -p "${CTX}/lib/${dest}"
    cp -a "${src}/." "${CTX}/lib/${dest}/"
  fi
}
_copy_tensile /opt/rocm/lib/hipblaslt/library hipblaslt/library
_copy_tensile /opt/rocm/lib64/hipblaslt/library hipblaslt/library
_copy_tensile /opt/rocm/hipblaslt/lib/library hipblaslt/library
_copy_tensile /opt/rocm/lib/rocblas/library rocblas/library
_copy_tensile /opt/rocm/lib64/rocblas/library rocblas/library
_copy_tensile /opt/rocm/rocblas/lib/library rocblas/library
if [ ! -d "${CTX}/lib/hipblaslt/library" ]; then
  echo "WARNING: no hipBLASLt Tensile library packed; GPU_IVF_FLAT search may fail on dim=128" >&2
fi

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

# Do not use grep -q in a pipe under pipefail: grep exits early, nm gets
# SIGPIPE (141), and a successful match looks like failure.
_nm_has() {
  local file="$1" pat="$2" mode="${3:-}"
  local out
  if [ "${mode}" = "-D" ]; then
    out="$(nm -D "${file}" 2>/dev/null | grep -F "${pat}" || true)"
  else
    out="$(nm "${file}" 2>/dev/null | grep -F "${pat}" || true)"
  fi
  [ -n "${out}" ]
}

_GLOG_FLAGREG_NEEDED=""

_load_glog_needed() {
  _GLOG_FLAGREG_NEEDED=""
  [ -f "${CTX}/lib/libglog.so.1" ] || return 0
  _GLOG_FLAGREG_NEEDED="$(nm -D "${CTX}/lib/libglog.so.1" 2>/dev/null \
    | awk '/ U / && /FlagRegisterer/ {print $NF}' | sort -u)"
  if [ -n "${_GLOG_FLAGREG_NEEDED}" ]; then
    echo "    glog undefined FlagRegisterer:"
    printf '%s\n' "${_GLOG_FLAGREG_NEEDED}" | sed 's/^/      /'
  fi
}

# True if every FlagRegisterer glog needs is DEFINED in this ELF.
_gflags_covers_glog() {
  local elf="$1" need defs
  [ -f "${elf}" ] || return 1
  if [ -z "${_GLOG_FLAGREG_NEEDED}" ]; then
    _nm_has "${elf}" '_ZN6gflags14FlagRegisterer' -D
    return $?
  fi
  while IFS= read -r need; do
    [ -n "${need}" ] || continue
    defs="$(nm -D "${elf}" 2>/dev/null | grep -F "${need}" | grep -v ' U ' || true)"
    if [ -z "${defs}" ]; then
      return 1
    fi
  done <<< "${_GLOG_FLAGREG_NEEDED}"
  return 0
}

_gflags_covers_glog_as_google() {
  local elf="$1" need gneed defs
  [ -f "${elf}" ] || return 1
  [ -n "${_GLOG_FLAGREG_NEEDED}" ] || return 1
  while IFS= read -r need; do
    [ -n "${need}" ] || continue
    gneed="${need//6gflags/6google}"
    defs="$(nm -D "${elf}" 2>/dev/null | grep -F "${gneed}" | grep -v ' U ' || true)"
    if [ -z "${defs}" ]; then
      return 1
    fi
  done <<< "${_GLOG_FLAGREG_NEEDED}"
  return 0
}

_rename_google_to_gflags() {
  local src="$1" dest="$2" map
  mkdir -p "$(dirname "${dest}")"
  map="${WORKDIR}/libshims/gflags_redefine.map"
  mkdir -p "${WORKDIR}/libshims"
  nm -D "${src}" 2>/dev/null | awk '{print $NF}' | grep '6google' | sort -u | while read -r _sym; do
    printf '%s %s\n' "${_sym}" "${_sym//6google/6gflags}"
  done > "${map}"
  [ -s "${map}" ] || return 1
  echo "    rename google:: -> gflags:: ($(wc -l < "${map}") symbols) from ${src}"
  objcopy --redefine-syms="${map}" "${src}" "${dest}"
  rm -f "${map}"
}

_install_gflags_elf() {
  local real="$1"
  local soname="" _n="" _src=""
  [ -f "${real}" ] || return 1
  _gflags_covers_glog "${real}" || return 1
  echo "    gflags covers glog: ${real}"
  _copy_so "${real}"
  cp -f "${real}" "${CTX}/lib/libgflags_gflagsns.so"
  soname="$(objdump -p "${real}" 2>/dev/null | awk '/SONAME/ {print $2; exit}')"
  if [ -z "${soname}" ]; then
    soname="$(readelf -d "${real}" 2>/dev/null | awk '/SONAME/ {gsub(/[\[\]]/,"",$NF); print $NF; exit}')"
  fi
  for _n in ${soname} libgflags.so.2.2 libgflags.so.2 libgflags.so \
            libgflags_nothreads.so.2.2.2 libgflags_nothreads.so.2.2 \
            libgflags_nothreads.so "$(basename "${real}")"; do
    [ -n "${_n}" ] || continue
    cp -f "${real}" "${CTX}/lib/${_n}"
  done
  for _src in "${CTX}/lib/libglog.so.1" "${CTX}/bin/milvus"; do
    [ -e "${_src}" ] || continue
    while IFS= read -r _n; do
      [ -n "${_n}" ] || continue
      cp -f "${real}" "${CTX}/lib/${_n}"
    done < <(readelf -d "${_src}" 2>/dev/null | awk '/NEEDED/ && /gflags/ {gsub(/[\[\]]/,"",$NF); print $NF}')
  done
  return 0
}

_try_gflags() {
  local src="$1" real dest
  [ -e "${src}" ] || return 1
  real="$(readlink -f "${src}")"
  [ -f "${real}" ] || return 1
  if _install_gflags_elf "${real}"; then
    return 0
  fi
  if _gflags_covers_glog_as_google "${real}"; then
    dest="${WORKDIR}/libshims/libgflags_gflagsns.so"
    if _rename_google_to_gflags "${real}" "${dest}" && _install_gflags_elf "${dest}"; then
      return 0
    fi
  fi
  return 1
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
_load_glog_needed

# Prefer Conan *shared* gflags (has FlagRegisterer<std::__cxx11::string>).
# The leftover .a shim only had the bool ctor — that is the previous image crash.
_gflags_ok=0
while IFS= read -r _cand; do
  [ -n "${_cand}" ] || continue
  echo "    try ${_cand}"
  if _try_gflags "${_cand}"; then
    _gflags_ok=1
    break
  fi
done < <(find \
    "${HOME}/.conan/data/gflags" "${HOME}/.conan2" \
    "${MILVUS_DIR}/internal/core/output/lib" "${MILVUS_DIR}/lib" \
    "${HIPVS_PREFIX}/lib" "${HIPVS_PREFIX}/lib64" \
    \( -name 'libgflags.so*' -o -name 'libgflags_nothreads.so*' \) \
    -type f 2>/dev/null | awk '!/\.a$/')

if [ "${_gflags_ok}" = "0" ]; then
  echo "    try leftover shim / rebuild from Conan .a"
  if _try_gflags "${WORKDIR}/libshims/libgflags_gflagsns.so"; then
    _gflags_ok=1
  fi
fi

if [ "${_gflags_ok}" = "0" ]; then
  mkdir -p "${WORKDIR}/libshims"
  while IFS= read -r _gflags_a; do
    [ -f "${_gflags_a}" ] || continue
    echo "    build gflags:: shim from ${_gflags_a}"
    _tmp_so="${WORKDIR}/libshims/libgflags_shim_tmp.so"
    _shim="${WORKDIR}/libshims/libgflags_gflagsns.so"
    g++ -shared -fPIC -o "${_tmp_so}" \
      -Wl,--whole-archive "${_gflags_a}" -Wl,--no-whole-archive \
      -lpthread -lrt
    if _gflags_covers_glog "${_tmp_so}"; then
      mv -f "${_tmp_so}" "${_shim}"
    elif _gflags_covers_glog_as_google "${_tmp_so}"; then
      _rename_google_to_gflags "${_tmp_so}" "${_shim}" || true
      rm -f "${_tmp_so}"
    else
      echo "    skip ${_gflags_a} (FlagRegisterer types do not cover glog)"
      rm -f "${_tmp_so}"
      continue
    fi
    if _install_gflags_elf "${_shim}"; then
      _gflags_ok=1
      break
    fi
  done < <(ls -1 \
    "${HOME}"/.conan/data/gflags/*/_/_/package/*/lib/libgflags_nothreads.a \
    "${HOME}"/.conan/data/gflags/*/_/_/package/*/lib/libgflags.a \
    "${HOME}"/.conan2/p/*/p/lib/libgflags_nothreads.a \
    "${HOME}"/.conan2/p/*/p/lib/libgflags.a \
    2>/dev/null || true)
fi

if [ -n "${_GLOG_FLAGREG_NEEDED}" ]; then
  _packed_covers=0
  for _f in "${CTX}/lib"/libgflags*.so*; do
    [ -f "${_f}" ] || continue
    if _gflags_covers_glog "${_f}"; then
      _packed_covers=1
      break
    fi
  done
  if [ "${_packed_covers}" != "1" ]; then
    echo "ERROR: packed gflags does not define every FlagRegisterer glog needs:" >&2
    printf '%s\n' "${_GLOG_FLAGREG_NEEDED}" | sed 's/^/  /' >&2
    echo "  nm -D on Conan shared gflags:" >&2
    find "${HOME}/.conan/data/gflags" -name 'libgflags_nothreads.so*' -type f 2>/dev/null \
      | while IFS= read -r _c; do
          echo "  -- ${_c}" >&2
          nm -D "${_c}" 2>/dev/null | grep FlagRegisterer | head -8 >&2 || true
        done
    exit 1
  fi
fi
if [ "${_gflags_ok}" = "1" ]; then
  echo "    FlagRegisterer pin ok"
elif [ -f "${CTX}/lib/libglog.so.1" ]; then
  echo "WARNING: no gflags that covers glog FlagRegisterer; container will likely die" >&2
fi

# Host ldd still sees /usr/lib spdlog, so a skip of apt libs looks "clean" on
# the box and then the container dies with libspdlog.so.1 not found.
echo "==> pin spdlog/fmt the host milvus loads"
while IFS= read -r _p; do
  [ -n "${_p}" ] && [ -e "${_p}" ] || continue
  _real="$(readlink -f "${_p}")"
  [ -f "${_real}" ] || continue
  echo "    ${_p} <- ${_real}"
  cp -f "${_real}" "${CTX}/lib/$(basename "${_real}")"
  cp -f "${_real}" "${CTX}/lib/$(basename "${_p}")"
  _sn="$(objdump -p "${_real}" 2>/dev/null | awk '/SONAME/ {print $2; exit}')"
  if [ -n "${_sn}" ]; then
    cp -f "${_real}" "${CTX}/lib/${_sn}"
  fi
done < <(ldd "${_bin}" 2>/dev/null | awk '/libspdlog|libfmt/ && /=> \// {print $3}')
if [ ! -e "${CTX}/lib/libspdlog.so.1" ]; then
  for _c in \
    "${MILVUS_DIR}/internal/core/output/lib"/libspdlog.so.1 \
    /usr/lib/x86_64-linux-gnu/libspdlog.so.1
  do
    [ -e "${_c}" ] || continue
    cp -f "$(readlink -f "${_c}")" "${CTX}/lib/libspdlog.so.1"
    echo "    fallback libspdlog.so.1 <- ${_c}"
    break
  done
fi
if [ ! -e "${CTX}/lib/libspdlog.so.1" ]; then
  echo "ERROR: libspdlog.so.1 not packed; container cannot start" >&2
  exit 1
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
