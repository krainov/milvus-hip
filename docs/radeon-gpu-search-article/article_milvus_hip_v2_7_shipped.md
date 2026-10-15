## 7. What we ported

Two GPU index types run end to end on this tree, Milvus v2.5.4.

`GPU_IVF_FLAT` splits the collection into clusters and keeps full-precision vectors inside each cluster. A query scans some of those clusters, not all of them. That scan count is nprobe.

`GPU_IVF_PQ` uses the same clusters, then compresses each vector with product quantization. Less GPU memory. Recall is usually a little lower.

`GPU_CAGRA` can search the whole index in the lab. Put a filter on the query, or delete a row, or set TTL, and do not use it here yet.

We have not taken this fork to Instinct. The amdgpu driver on the AMD box was 6.14.14; the NVIDIA box ran driver 595.71.05.
