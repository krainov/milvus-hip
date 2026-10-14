## 2. Choosing a vector database

We compared the usual tools against three needs: a client API, collections the database keeps on disk, and GPU nearest-neighbour search.

| Tool | What it is | Fit for this job |
|------|------------|------------------|
| [FAISS](https://github.com/facebookresearch/faiss) | ANN library | No server. No database-managed collections. |
| [pgvector](https://github.com/pgvector/pgvector) | PostgreSQL extension | Search runs on the CPU. |
| [Qdrant](https://qdrant.tech/) | Vector database | A database, but not the Milvus GPU stack. |
| [Milvus](https://milvus.io/) | Vector database | GPU indexes through Knowhere, its search engine. About 46,000 GitHub stars. |

We chose Milvus. It is the most popular open-source vector database for retrieval-augmented generation (RAG) and approximate nearest-neighbour (ANN) search. It already ships GPU indexes. We only looked at tools we could run on our own machine.
