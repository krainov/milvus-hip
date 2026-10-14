## 3. The gap

Stock Milvus GPU search runs on CUDA. The Python client does not call the GPU library. The path is [pymilvus](https://milvus.io/docs) to the Milvus server to [Knowhere](https://github.com/milvus-io/knowhere) to [cuVS](https://docs.rapids.ai/api/cuvs/stable/) / RAFT. cuVS runs on CUDA. NVIDIA keeps one stack on every tier: a consumer GeForce RTX card and a datacentre accelerator run the same GPU search path.

```
pymilvus
    → Milvus server
        → Knowhere
            → GPU_IVF_FLAT / GPU_IVF_PQ
                → NVIDIA: RAFT / cuVS
                → AMD:    hipRAFT / hipVS
```

AMD’s counterpart is [hipVS](https://github.com/ROCm-DS/hipVS), built on [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html). hipVS 0.1.0 covers Instinct cards in the datacentre. It does not cover consumer Radeon. The workstation in the example has an RX 7900 XTX (`gfx1100`). On that card, Milvus never reaches a GPU index. A sealed `GPU_IVF_FLAT` or `GPU_IVF_PQ` does not build, load, or search. Knowhere stays on the CPU.

CPU Milvus already ran on the same box. Neighbour IDs came back. The lookup was slower, and the GPU did none of it. We do not have a published CPU-versus-GPU QPS table on the same GIST-1M nightly job, so this section does not quote a CPU baseline. The later plots compare two GPU builds.

A CUDA Milvus GPU image does not close this. Pulling it on a Radeon still leaves GPU indexes on NVIDIA. GPU search on this card meant building Knowhere against hipVS from source, and getting hipVS itself to run on `gfx1100`, not only on Instinct.

The gap was GPU approximate nearest-neighbour search inside Milvus on a consumer Radeon. CPU search existed. GPU search did not.
