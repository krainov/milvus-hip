## 1. A local RAG workstation (example)

Picture a team building a local RAG assistant — a language model that answers from their own files — on a workstation with a Radeon card. They work from the company's own documents: manuals, tickets, design notes. Policy says that text, embeddings, and query logs must not leave the site. A cloud vector store is not allowed. The assistant still has to answer while someone waits, so nearest-neighbour lookup over those embeddings has to run on that workstation, on the GPU already in the machine.

CPU search can do the job at small scale. As the collection grows, latency climbs, and the team wants the lookup on the GPU. The card is a consumer Radeon, not a datacentre accelerator. They do not get to pick a different GPU for this trial. They need GPU nearest-neighbour search on the hardware they have, with the data staying local.

The next choice is which vector database to run. It needs a client API, indexes on disk, and a GPU search path.
