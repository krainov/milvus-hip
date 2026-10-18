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
exec /tini -- "$@"
