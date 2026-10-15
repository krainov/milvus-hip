## 8. How we measured

The AMD box has 64 GB of DDR5. That was never the limit. The limit was GPU memory, 24 GB versus 16 GB, so we picked GIST-1M: a million vectors, 960 dimensions, about 3.8 GB. It fits both cards. If the collection is larger than the GPU, these plots are not that collection.

Both machines ran the same queries and the same index settings, every night. We wanted a thousand repeats of the same sweep, not one lucky run. The line is median queries per second (QPS) versus nprobe. Whiskers are one standard deviation across those thousand sweeps.

A sweep that fails the recall gate against exact kNN — the same gate as the NVIDIA build — is thrown away. So is a sweep whose neighbour IDs diverge from the NVIDIA run.

Every run stores a dataset hash and the build commit. If a number looks odd next month, we can open that night's binary. The live dashboard shows correctness, speed, and that trail in one view. When an older one-shot harness table disagrees, the dashboard wins.
