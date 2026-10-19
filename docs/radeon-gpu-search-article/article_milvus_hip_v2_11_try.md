## 11. Try it on your machine

The public HIP line is Milvus v2.5.4. The closed agent pipeline is not published, so it is not in this list.

- Milvus: [krainov/milvus-hip](https://github.com/krainov/milvus-hip), branch `v2.5.4`
- Knowhere: [krainov/knowhere-hip](https://github.com/krainov/knowhere-hip), branch `2.5`
- Build and benchmark: [konkolchin/ann-harness-amd](https://github.com/konkolchin/ann-harness-amd)

This preview has no HIP container. The Docker image `milvus-standalone` is a CUDA build. On a Radeon it does not run GPU search. Build on Ubuntu 22.04 with ROCm 7.0.2. The card we validated is an RX 7900 XTX (`gfx1100`).

```bash
git clone https://github.com/konkolchin/ann-harness-amd.git
cd ann-harness-amd
mkdir -p data
wget -c https://ann-benchmarks.com/gist-960-euclidean.hdf5 -O data/gist-960-euclidean.hdf5
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export MILVUS_WITH_HIP=1
export KNOWHERE_WITH_HIP=1
# KNOWHERE_DIR: clone of krainov/knowhere-hip, branch 2.5
# HIPVS_PREFIX: hipVS install from scripts/check_rocmds_gfx1100.sh
bash scripts/build_milvus_layer3.sh
bash scripts/run_milvus_gpu_smoke.sh
bash scripts/run_milvus_layer4_gist_flat.sh
bash scripts/run_milvus_layer4_gist_pq.sh
```

`build_milvus_layer3.sh` clones Milvus v2.5.4, applies the HIP patches, and links Knowhere when `KNOWHERE_DIR` points at the `2.5` tree. `scripts/check_rocmds_gfx1100.sh` is the hipVS check for this GPU. Smoke checks a sealed `GPU_IVF_FLAT` search on `127.0.0.1:19530`. The two GIST scripts sweep nprobe 8, 16, 32, 64, and 128. They are short runs. The figures above are the nightly median of 1,000 sweeps. If a short run and that median disagree, we publish the nightly median.
