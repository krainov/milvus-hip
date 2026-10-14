## 1. A local RAG workstation (example)

The next two paragraphs are an example, not a named customer.

Picture a team building a local RAG assistant on a workstation with a Radeon card. The corpus is internal: manuals, tickets, design notes. Policy says that text, embeddings, and query logs must not leave the site. A cloud vector store is off the table. The assistant still has to answer while someone waits, so nearest-neighbour lookup over those embeddings has to run in the same room, on the GPU already in the case.

CPU search can do the job at small scale. As the collection grows, latency climbs and the GPU is the obvious next step. The card on this desk is a consumer Radeon, not a datacentre accelerator. The team does not get to pick a different GPU for the pilot. They need GPU nearest-neighbour search on the hardware they have, with the data staying local.

The next choice is which vector database to run. It has to speak a normal client API, keep indexes on disk, and already have a GPU search path that is used in production RAG stacks — or the pilot will stall on plumbing instead of on retrieval quality.
