# mHC Kernel Benchmarking

This repository is based on the [TileKernels mHC implementation](https://github.com/deepseek-ai/TileKernels/tree/main/tile_kernels/mhc) and extends the benchmark coverage with mHC kernels from the Megatron-LM dev branch and the [mhc_bench](https://github.com/kainzhong/mhc_bench/tree/main) Triton implementation.

The mHC benchmark compares three kernel sources:

- **TE (Triton)** — Triton kernels from `mhc_bench`
- **Megatron-LM (cuTile + Triton)** — fused mHC kernels from the Megatron-LM dev branch
- **TileKernels (TileLang)** — TileLang kernels from TileKernels

Autotuning is enabled for the TileKernels mHC kernels and for the Megatron-LM dev-branch mHC kernels before collecting the benchmark results below.

### GB200 Cluster Results

All latencies are measured on a GB200 cluster after autotuning. Units are microseconds. These benchmark results are collected with the scripts in this repository and report Nsight GPU kernel time only, excluding GPU bubbles and kernel launch latency.

#### `s*b=4096, h=7168, n=4`


| Kernel            | TE (Triton) | Megatron-LM (cuTile + Triton) | TileKernels (TileLang) | Hint                                                                 |
| ----------------- | ----------- | ----------------------------- | ---------------------- | -------------------------------------------------------------------- |
| pre_mix (fwd)     | 44          | 42                            | 41                     |                                                                      |
| pre_mix (bwd)     | 85          | 83                            | 112                    |                                                                      |
| mhc_post (fwd)    | 74          | 80                            | 78                     |                                                                      |
| mhc_post (bwd)    | 187         | 183                           | 200                    |                                                                      |
| proj+reduce (fwd) | 81          | 61                            | 96                     | So many small kernels in TE's proj + reduce impl                     |
| proj+reduce (bwd) | 170         | 169                           | 425                    | Try to apply auto-tune for tilelang proj+reduce kernels, but failed. |
| sinkhorn (fwd)    | 4.5         | 4.6                           | 7.8                    |                                                                      |
| sinkhorn (bwd)    | 9.5         | 7                             | 12.3                   |                                                                      |
| fwd sum           | 203.5       | 187.6                         | 222.8                  |                                                                      |
| bwd sum           | 451.5       | 442                           | 749.3                  |                                                                      |


#### `s*b=4096, h=4096, n=4`

Sinkhorn is omitted because it is the same as the `h=7168` case. Units are microseconds.


| Kernel                 | TE (Triton) | Megatron-LM (cuTile + Triton) | TileKernels (TileLang) | Hint |
| ---------------------- | ----------- | ----------------------------- | ---------------------- | ---- |
| pre_mix (fwd)          | 25          | 23                            | 26                     |      |
| pre_mix (bwd)          | 49          | 55                            | 66                     |      |
| mhc_post (fwd)         | 43          | 47                            | 47                     |      |
| mhc_post (bwd)         | 110         | 111                           | 123                    |      |
| proj+reduce (fwd)      | 46          | 41                            | 57                     |      |
| proj+reduce (bwd)      | 122         | 99                            | 370                    |      |
| fwd sum (w/o sinkhorn) | 114         | 111                           | 130                    |      |
| bwd sum (w/o sinkhorn) | 281         | 265                           | 559                    |      |


#### `s*b=16384, h=7168, n=4`


| Kernel            | TE (Triton) | Megatron-LM (cuTile + Triton) | TileKernels (TileLang) | Hint |
| ----------------- | ----------- | ----------------------------- | ---------------------- | ---- |
| pre_mix (fwd)     | 165         | 167                           | 165                    |      |
| pre_mix (bwd)     | 330         | 314                           | 427                    |      |
| mhc_post (fwd)    | 292         | 311                           | 299                    |      |
| mhc_post (bwd)    | 737         | 651                           | 761                    |      |
| proj+reduce (fwd) | 187         | 209                           | 321                    |      |
| proj+reduce (bwd) | 560         | 592                           | 1634                   |      |
| sinkhorn (fwd)    | 7.2         | 11                            | 15.3                   |      |
| sinkhorn (bwd)    | 11.8        | 18.6                          | 24.1                   |      |
| fwd sum           | 651.2       | 698                           | 800.3                  |      |
| bwd sum           | 1638.8      | 1575.6                        | 2846.1                 |      |


## Nsight Systems Capture

```bash
./capture_mhc_timeline.sh smoke \
  --source tilelang,megatron_lm,mhc_bench_triton \
  --scope kernels \
  --shape 4096,1,7168
```

