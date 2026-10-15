## 1. A local RAG workstation (example)

Picture a team building a local RAG assistant — a language model that answers from their own files — on a workstation with a Radeon card. They work from the company's own documents: manuals, tickets, design notes. Policy says that text, embeddings, and query logs must not leave the site. A cloud vector store is not allowed.

The assistant still has to answer while someone waits. Nearest-neighbour lookup over those embeddings has to run in the same room, on the GPU already in the machine. CPU search is fine while the collection is small. As it grows, latency climbs. The card is a consumer Radeon, not a datacentre accelerator, and nobody is swapping it for the pilot.

Then they still have to pick a vector database.
