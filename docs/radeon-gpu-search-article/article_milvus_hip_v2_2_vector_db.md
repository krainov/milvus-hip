## 2. Choosing a vector database

That choice is [Milvus](https://milvus.io/). It is the most popular open-source vector database for retrieval-augmented generation (RAG) and approximate nearest-neighbour (ANN) search: about 46,000 GitHub stars. On NVIDIA it already has GPU indexes through [Knowhere](https://github.com/milvus-io/knowhere), the search engine inside Milvus. `GPU_IVF_FLAT` keeps full vectors inside inverted-file clusters. `GPU_IVF_PQ` compresses them with product quantization.

Other tools existed. [FAISS](https://github.com/facebookresearch/faiss) is a library that runs inside a process, not a database with a server and on-disk collections. [pgvector](https://github.com/pgvector/pgvector) adds vector search to PostgreSQL and runs on the CPU. [Qdrant](https://qdrant.tech/) is another vector database. We chose Milvus for that popularity and for its GPU path: Knowhere calls NVIDIA [cuVS](https://docs.rapids.ai/api/cuvs/stable/). That is the path we needed on the Radeon.
