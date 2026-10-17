#!/bin/sh
# Optional jemalloc (present on stock Milvus GPU images; HIP builds may omit it).
if [ -f /milvus/lib/libjemalloc.so ]; then
  export LD_PRELOAD="/milvus/lib/libjemalloc.so${LD_PRELOAD:+:$LD_PRELOAD}"
  export MALLOC_CONF="${MALLOC_CONF:-background_thread:true}"
fi
export LD_LIBRARY_PATH="/milvus/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /tini -- "$@"
