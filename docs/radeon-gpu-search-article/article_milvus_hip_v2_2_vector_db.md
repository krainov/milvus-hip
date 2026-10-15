## 2. Choosing a vector database

We looked at the names people actually put in a RAG stack. We only compared tools we could run on our own machine.

| Tool | What it is | Fit for this job |
|------|------------|------------------|
| [FAISS](https://github.com/facebookresearch/faiss) | ANN library | No server. No database-managed collections. |
| [pgvector](https://github.com/pgvector/pgvector) | PostgreSQL extension | Search runs on the CPU. |
| [Qdrant](https://qdrant.tech/) | Vector database | A database; we needed Milvus’s GPU-index path. |
| [Milvus](https://milvus.io/) | Vector database | GPU indexes through Knowhere, its search engine. |

We chose [Milvus](https://milvus.io/): it is the most popular open-source vector database for RAG and ANN search — about 46,000 GitHub stars — and it already ships GPU indexes.
