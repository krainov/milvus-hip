## 1. A local RAG workstation (example)

Picture a team building a local RAG (Retrieval-Augmented Generation) assistant — a chatbot that uses a large language model (LLM) that answers from the company's own files. The team works from manuals, tickets, and design notes. Policy says that text, embeddings, and query logs must not leave the site: no cloud services.

The RAG assistant has to answer while the user waits. Nearest-neighbour lookup over those embeddings has to run on the workstation they already have. CPU search is fine while the embeddings collection is small. As it grows, latency climbs, so the heavy lookup has to run on the GPU. The card in the team's workstation is a consumer Radeon, not a datacentre accelerator, and this trial has to run on that card. They still have to pick a vector database.
