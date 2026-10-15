## 2. Choosing a vector database

We looked at the names people actually put in a RAG stack.

| Tool | What it is | Fit for this job |
|------|------------|------------------|
| [FAISS](https://github.com/facebookresearch/faiss) | ANN library | No server. No database-managed collections. |
| [pgvector](https://github.com/pgvector/pgvector) | PostgreSQL extension | Search runs on the CPU. |
| [Qdrant](https://qdrant.tech/) | Vector database | A database, but not the Milvus GPU stack. |
| [Milvus](https://milvus.io/) | Vector database | GPU indexes through Knowhere, its search engine. |

We chose [Milvus](https://milvus.io/). It is the most popular open-source vector database for retrieval-augmented generation (RAG) and approximate nearest-neighbour (ANN) search — about 46,000 GitHub stars — and it already ships GPU indexes. We only looked at tools we could run on our own machine.
