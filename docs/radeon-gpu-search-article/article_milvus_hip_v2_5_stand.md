## 5. Two machines

The port ran on the AMD workstation first. The CPU in that box is a Ryzen 7 7700X. We did not need NVIDIA in the building to make search work.

After it did work, we added a GeForce RTX 4080 (16 GB) so we could compare the two builds and publish. The cards are the same generation: they shipped about four weeks apart in late 2022. They are not the same chip. The Radeon has 24 GB of memory; the 4080 has 16 GB. That gap matters later, when the dataset has to fit both.

The NVIDIA box is Ubuntu 22.04.5 as well, with an Intel CPU (16 logical CPUs; the exact model was not in the lab notes) and about 61 GiB of RAM as the OS sees it. Its Milvus binary was built with CUDA 11.8. CUDA 12.8 showed up only in a separate cuVS container, not in that binary.
