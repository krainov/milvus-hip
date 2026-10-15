## 3. The gap

Official Milvus can run GPU search. That search runs on CUDA. We turned the GPU index on. On AMD hardware it did not come up. CPU search still answered. The GPU did none of the work.

The official GPU Docker image was the obvious next try. It is an NVIDIA CUDA image. Pulling it on a Radeon does not give GPU search.

The reason is the call path. The Python client, [pymilvus](https://milvus.io/docs), does not call the GPU library. It talks to the Milvus server. The server passes the request to [Knowhere](https://github.com/milvus-io/knowhere). Knowhere calls [cuVS](https://docs.rapids.ai/api/cuvs/stable/), NVIDIA's GPU vector-search library, which sits on RAFT. That whole GPU branch needs CUDA.

```
pymilvus → Milvus server → Knowhere
    NVIDIA (official): RAFT / cuVS     (CUDA)
    AMD (this port):   hipRAFT / hipVS (HIP)
```

AMD's counterpart is [hipVS](https://github.com/ROCm-DS/hipVS), on [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html). hipVS uses hipRAFT the way cuVS uses RAFT. Version 0.1.0 supports AMD Instinct cards in the datacentre. It does not support consumer Radeon.

The workstation in the example has an RX 7900 XTX (`gfx1100`). Official Milvus could not run a GPU index on that card.
