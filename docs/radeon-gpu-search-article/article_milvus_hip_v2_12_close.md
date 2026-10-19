## 12. Back on the workstation

The team in the example can keep the documents on site. Embeddings and query logs stay on the machine. The client is still pymilvus, pointed at a Milvus server in that room. The server is this HIP build. Knowhere calls hipVS. On the Radeon already in the workstation, the GPU index builds, loads, and searches.

On GIST-1M, recall@10 matches the NVIDIA build. Uncompressed `GPU_IVF_FLAT` is faster on the RX 7900 XTX at every nprobe we ran: about +15–30% at nprobe 8, and about 1.3× at nprobe 128. Compressed `GPU_IVF_PQ` is mixed. The nightly median is +36% at nprobe 8 and −22% at nprobe 128. Inside one mode the 4080 is ahead, about 0.87× to 0.65×.
