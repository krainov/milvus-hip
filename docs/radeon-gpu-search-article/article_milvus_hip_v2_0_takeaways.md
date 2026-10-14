# Bringing Milvus GPU search to Radeon: a port, a benchmark, and what we learned

**Key takeaways.** We ported Milvus GPU approximate nearest-neighbour (ANN) search from CUDA to HIP and ran it end to end on a Radeon RX 7900 XTX. On the same GIST-1M recipes, neighbour IDs match the NVIDIA build. Uncompressed `GPU_IVF_FLAT` is faster on the 7900 XTX than on an RTX 4080 at every nprobe we ran — **+26%** queries per second at nprobe 8 (nprobe is how many index clusters a query scans). Compressed `GPU_IVF_PQ` is **+36%** at nprobe 8 and **−22%** at nprobe 128.

![GPU_IVF_FLAT on GIST-1M: median QPS vs nprobe, Radeon RX 7900 XTX vs GeForce RTX 4080](fig_gist_flat_qps.jpg)
