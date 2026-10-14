## 3. The gap

Official Milvus can run GPU search. That search runs on CUDA. On AMD hardware, the official GPU path is not available. CPU search still works.

The official GPU Docker image does not change that. It is built for NVIDIA CUDA. Running that image on a Radeon does not give GPU search.

The reason is the call path. The Python client, [pymilvus](https://milvus.io/docs), does not call the GPU library. It talks to the Milvus server. The server passes the request to [Knowhere](https://github.com/milvus-io/knowhere). Knowhere calls [cuVS](https://docs.rapids.ai/api/cuvs/stable/), NVIDIA's GPU vector-search library. cuVS uses RAFT, a lower-level NVIDIA GPU library. cuVS needs CUDA.

```
pymilvus → Milvus server → Knowhere
    NVIDIA (official): RAFT / cuVS     (CUDA)
    AMD (this port):   hipRAFT / hipVS (HIP)
```

AMD's counterpart is [hipVS](https://github.com/ROCm-DS/hipVS), on [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html), AMD's GPU programming interface. hipVS uses hipRAFT for the lower-level GPU algorithms. hipVS 0.1.0 supports AMD Instinct datacentre GPUs. It does not support consumer Radeon.

The workstation in the example has an RX 7900 XTX (`gfx1100`). Official Milvus could not run a GPU index on that card. To get GPU search, we built Knowhere against hipVS from source and made hipVS run on `gfx1100`.
