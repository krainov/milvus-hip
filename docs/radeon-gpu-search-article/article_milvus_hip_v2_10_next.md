## 10. What is next

Three checks are still open.

Compressed search at high nprobe is the tuning job. This HIP port kept the NVIDIA PQ kernels and ran them on the Radeon. The same-mode gap, about **0.87×** at low nprobe and about **0.65×** at high nprobe, is the gap we would try to close. That work is kernel tuning for this GPU, a separate step from the port. A faster kernel that fails the recall gate is not a result we would keep.

`GPU_CAGRA` is the next index. Unfiltered search over a CAGRA graph already runs in the lab on this Radeon, with `build_algo` set to `NN_DESCENT`. A filter, a delete, or TTL still needs the bitset path. That path is unfinished. Until it passes the same recall gate as the NVIDIA build, CAGRA stays off the supported list. When a query must skip rows, use `GPU_IVF_FLAT` or `GPU_IVF_PQ`.

The third check is other GPUs. These plots are one RX 7900 XTX (`gfx1100`) and one RTX 4080. We have not run the fork on a newer Radeon, a newer GeForce, or an Instinct card. The packer fix is for a wavefront of 32. Instinct GPUs use a wavefront of 64, and stock [hipVS](https://github.com/ROCm-DS/hipVS) already targets those cards. Bringing this Milvus tree up there is a separate pass on the same software line.
