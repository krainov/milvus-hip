## 3. The gap

Official Milvus can run GPU search, but only on CUDA. We turned GPU indexing on. On our AMD card it did not come up. CPU search still answered. The GPU did none of the work.

The official GPU Docker image was the obvious next try. It is built for NVIDIA CUDA, so it was never going to help on a Radeon.

The reason is the call path. The Python client, [pymilvus](https://milvus.io/docs), does not call the GPU library. It talks to the Milvus server. The server passes the request to [Knowhere](https://github.com/milvus-io/knowhere). Knowhere calls [cuVS](https://docs.rapids.ai/api/cuvs/stable/), NVIDIA's GPU vector-search library, which sits on RAFT. That whole GPU branch needs CUDA.
