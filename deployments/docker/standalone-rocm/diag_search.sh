#!/usr/bin/env bash
# After GPU_IVF_FLAT search fails with "argument not found".
# Prints C++ logs, which libfmt is loaded, and CPU vs GPU search on a tiny set.
set -euo pipefail
URI="${MILVUS_URI:-http://127.0.0.1:19530}"

echo "==> C++ / knowhere lines (not Go proxy stack)"
docker logs milvus-hip-standalone 2>&1 | grep -aE \
  'C Runtime Exception|failed to search: config|format_error|N3fmt|PanicInfo|hipError|InvalidDevice|KnowhereException' \
  | tail -30 || true

echo "==> libfmt / libspdlog in the container"
docker exec milvus-hip-standalone sh -c \
  'ldd /milvus/bin/milvus | grep -E "fmt|spdlog" || true; ls -l /milvus/lib/libfmt* /milvus/lib/libspdlog* 2>/dev/null || true'

echo "==> CPU IVF_FLAT vs GPU_IVF_FLAT (8-d, 200 rows)"
python3 - "$URI" <<'PY'
import random, sys
from pymilvus import MilvusClient, DataType
uri = sys.argv[1]
c = MilvusClient(uri)

def trial(name, index_type):
    if c.has_collection(name):
        try:
            c.drop_collection(name)
        except Exception as exc:
            print(f"  drop {name} skipped: {exc}")
            name = name + "_n"
    schema = c.create_schema(auto_id=False, enable_dynamic_fields=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=8)
    c.create_collection(name, schema=schema)
    rows = [{"id": i, "vec": [random.random() for _ in range(8)]} for i in range(200)]
    c.insert(name, rows)
    c.flush(name)
    ip = c.prepare_index_params()
    ip.add_index(field_name="vec", index_type=index_type, metric_type="L2", params={"nlist": 16})
    c.create_index(name, ip)
    c.load_collection(name)
    try:
        res = c.search(
            collection_name=name,
            data=[[0.1] * 8],
            anns_field="vec",
            search_params={"metric_type": "L2", "params": {"nprobe": 4}},
            limit=3,
        )
        print(f"  {index_type} OK id={res[0][0]['id']}")
    except Exception as exc:
        print(f"  {index_type} FAIL {exc}")

trial("diag_cpu_ivf", "IVF_FLAT")
trial("diag_gpu_ivf", "GPU_IVF_FLAT")
PY
