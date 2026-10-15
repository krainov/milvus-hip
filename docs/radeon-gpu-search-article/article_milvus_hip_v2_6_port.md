## 6. How we ported it

We brought the stack up one layer at a time. Skip a layer and the next one can pass for the wrong reason.

| Layer | What we needed | Pass |
|-------|----------------|------|
| 0 | ROCm 7.0.2 and CPU Milvus on Ubuntu 22.04.5 | `gfx1100` visible; Standalone answered |
| 1 | hipRAFT / hipVS | Build; neighbour tests |
| 1.5 | IVF packer on this GPU | Float IVF tests 98/98 after `kIndexGroupSize=32` |
| 2 | Knowhere linked to hipVS | GPU L2 (Euclidean) search tests green |
| 3 | Milvus from source | Sealed `GPU_IVF_*` build / load / search |
| 4 | Fair benches | Same recall gate as CUDA |

A sealed index is a segment closed to new writes. That is where Milvus actually builds the GPU index. Growing segments can stay on the CPU and fool you.

A HIP port is not a rewrite of the kernels for RDNA3. We ran the existing algorithms through HIP. We only changed inner loops when the answers were wrong.

We tried two ways. One part of the team sat in the IDE with a coding assistant, edited C++, ran the gates, and decided which numbers could leave the lab. The packer fix and the nightly plots in this article come from that line.

Another part of the team ran a closed agent pipeline: adapt hipVS and hipRAFT, build Knowhere, run tests, pause for a person at each gate. Almost no hand-written C++. That pipeline is not on public GitHub yet. Its dashboard needs a rerun, so those numbers are not in the plots below.

![Closed agent pipeline for the HIP port: ordinary build steps, agent, gates, and human review](milvus_pipeline_architecture_v3.png)

Grey is an ordinary build or data step. Orange is the agent. Green is a gate. Purple is a person.

Search came back. The process looked healthy. The neighbour lists were empty, or the IDs belonged to someone else. The IVF packer still grouped work in units of 64 — Instinct's wavefront. This `gfx1100` path needed groups of 32, like an NVIDIA warp. `kIndexGroupSize = 32` and `USE_WARPSIZE_32`. After that, the float IVF tests were 98/98.

The build was the other wall. HIP flags leaked onto the host C++ compiler and had to be stripped. We also left an official `milvus-standalone` container running once and spent an hour debugging the wrong `libcuvs`.

We tried a “smarter” product-quantization score loop. Queries per second did not move. We reverted it. On the library side only, 512 threads per block helped PQ on this GPU. That number is not the Milvus QPS later.
