## 5. Two machines

The port ran on the AMD workstation first. The CPU in that box is a Ryzen 7 7700X.

After it did work, we added a GeForce RTX 4080 (16 GB) so we could compare the two builds and publish. The cards are the same generation: they shipped about four weeks apart in late 2022. They are not the same chip. The Radeon has 24 GB of memory; the 4080 has 16 GB. That split is not why QPS sometimes favours AMD. The IVF indexes we timed sat well under 16 GB on both cards. Spare VRAM does not raise queries per second.

Both boxes run Ubuntu 22.04.5. The NVIDIA machine has an Intel CPU (16 logical CPUs). Its Milvus binary was built with CUDA 11.8.
