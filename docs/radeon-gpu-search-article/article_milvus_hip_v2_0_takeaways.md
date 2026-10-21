# Bringing Milvus GPU search to Radeon: a port, a benchmark, and what we learned

**Key takeaways.** We ported Milvus GPU approximate nearest-neighbour (ANN) search from CUDA to HIP and ran it end to end on a Radeon RX 7900 XTX. On the GIST-1M recipes we checked, recall@10 matches the NVIDIA build (0.9910, 0.9918, and 0.9909). Uncompressed `GPU_IVF_FLAT` has a higher median queries per second on the 7900 XTX than on an RTX 4080 at every nprobe from 8 to 128 (nprobe is how many index clusters a query scans), and the gap is about 1.3× at nprobe 128. Compressed `GPU_IVF_PQ` is not a clean comparison: the two median curves cross near nprobe 32.

![GPU_IVF_FLAT on GIST-1M: median QPS vs nprobe, Radeon RX 7900 XTX vs GeForce RTX 4080](fig_gist_flat_qps.jpg)
