# How we made Milvus vector search work on an RX 7900 XTX

We needed a vector database for retrieval-augmented generation (RAG) on a PC we already had. In RAG, documents become embeddings — lists of floating-point numbers — and a question becomes an embedding too. The database then finds which stored vectors sit nearest to the question vector. That lookup is approximate nearest-neighbour search (ANN). People keep those vectors in a vector database. We wanted the heavy lookup on the GPU, inside that database, not in a side library.

We chose [Milvus](https://milvus.io/) because it is the most popular open-source vector database for RAG and ANN (about 46,000 GitHub stars). That is not the same as “there were no other tools.” FAISS is a library. pgvector is a PostgreSQL extension. Cloud vector stores were not the target. We needed the stack that RAG systems actually run, with GPU indexes.

The GPU in that PC was a consumer Radeon RX 7900 XTX (`gfx1100`). Stock Milvus GPU indexes use NVIDIA CUDA and [cuVS](https://docs.rapids.ai/api/cuvs/stable/). AMD’s counterpart is [hipVS](https://github.com/ROCm-DS/hipVS); version 0.1.0 covers Instinct cards in the datacentre. It does not cover this Radeon. [CPU Milvus](https://milvus.io/docs) already ran on the same box. GPU search did not.

The job, then, was to run Milvus ANN on that Radeon through [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html): adapt hipVS / hipRAFT, rebuild [Knowhere](https://github.com/milvus-io/knowhere) (the Milvus search engine) so it links hipVS instead of cuVS, and get sealed GPU indexes to build, load, and search. We did that work on the AMD machine first. Two index types came through end to end: `GPU_IVF_FLAT` (vectors stay uncompressed inside IVF clusters) and `GPU_IVF_PQ` (product quantization: less GPU memory, some recall cost). The line is Milvus v2.5.4.

After GPU search worked on the Radeon, we obtained a second machine with an NVIDIA GPU so we could compare the two ports and publish numbers.
