#!/bin/sh
# Same preload the host uses: Conan glog looks up gflags::FlagRegisterer.
if [ -f /milvus/lib/libgflags_gflagsns.so ]; then
  export LD_PRELOAD="/milvus/lib/libgflags_gflagsns.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi
if [ -f /milvus/lib/libjemalloc.so ]; then
  export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/milvus/lib/libjemalloc.so"
  export MALLOC_CONF="${MALLOC_CONF:-background_thread:true}"
fi
export LD_LIBRARY_PATH="/milvus/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -d /milvus/lib/hipblaslt/library ]; then
  export HIPBLASLT_TENSILE_LIBPATH="/milvus/lib/hipblaslt/library"
fi
if [ -d /milvus/lib/rocblas/library ]; then
  export ROCBLAS_TENSILE_LIBPATH="/milvus/lib/rocblas/library"
fi
if [ -z "${HIPBLASLT_TENSILE_LIBPATH:-}" ] && [ -d /opt/rocm/lib/hipblaslt/library ]; then
  export HIPBLASLT_TENSILE_LIBPATH="/opt/rocm/lib/hipblaslt/library"
fi
if [ -z "${ROCBLAS_TENSILE_LIBPATH:-}" ] && [ -d /opt/rocm/lib/rocblas/library ]; then
  export ROCBLAS_TENSILE_LIBPATH="/opt/rocm/lib/rocblas/library"
fi
exec /tini -- "$@"
