## 3. The gap

Stock Milvus GPU search needs CUDA. The Python client does not call cuVS. The path is [pymilvus](https://milvus.io/docs) to the Milvus server to Knowhere to [cuVS](https://docs.rapids.ai/api/cuvs/stable/) (NVIDIA's GPU vector-search library, built on RAFT).

```
pymilvus → Milvus server → Knowhere
    NVIDIA (stock):  RAFT / cuVS     (CUDA)
    AMD (this port): hipRAFT / hipVS (HIP)
```

AMD's counterpart is [hipVS](https://github.com/ROCm-DS/hipVS), on [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html), AMD's CUDA-like GPU API. hipRAFT sits under hipVS the way RAFT sits under cuVS. hipVS 0.1.0 supports Instinct datacentre GPUs. It does not support consumer Radeon.

The workstation in the example has an RX 7900 XTX (`gfx1100`). With the stock stack, Milvus could not build, load, or search `GPU_IVF_FLAT` or `GPU_IVF_PQ` on that card. CPU Milvus still ran. The GPU did none of the lookup. We do not report a CPU-versus-GPU QPS (queries per second) figure here.

The NVIDIA GPU Docker image does not help: it needs an NVIDIA GPU. GPU search on this card meant building Knowhere against hipVS from source, and getting hipVS itself to run on `gfx1100`.
