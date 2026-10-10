# CUDA triangle counting

GPU triangle counting on SNAP graphs (Wiki-Vote, email-Enron, as-Skitter), with CPU baselines
(serial + OpenMP), exact count verification and a results table. Reuses the degree-ordered
orientation idea from my Densest Subgraph project.

## Method
Edges are oriented from the lower (degree, id) endpoint to the higher one, so each triangle is
found once: `triangles = sum over oriented edges (u,v) of |N+(u) ∩ N+(v)|`.

## Layout
- `src/graph.h` loader, dedup, orientation, CSR, CPU counters
- `src/cpu_tc.cpp` CPU baselines
- `src/kernels.cuh` CUDA kernels v1-v9, `src/gpu_tc.cu` driver (timing, verification)
- `src/densest.h` densest-subgraph CPU references (batch peeling, greedy peeling), the host peeling loop; `src/cpu_densest.cpp` CPU-only runner; `src/gpu_densest.cu` GPU driver (stage 5)
- `scripts/` data download, synthetic test generator, benchmark runner, table/chart/bandwidth-estimate scripts
- `emu/` CPU thread emulator of the kernels, tested against brute force (`make test`, no GPU needed)
- `results/` `results.csv`, raw Colab logs, charts; `docs/analysis.md` full write-up
- `colab_run.ipynb` run everything on a free Colab GPU

## Build and run
    make cpu                      # CPU only (no GPU needed)
    make ARCH=sm_75               # CPU + GPU (T4)
    bash scripts/download_data.sh
    bash scripts/run_bench.sh

## Results
GPU: Tesla T4 (Colab, Turing 2018), CPU: 2 cores on the same Colab VM. Each cell is the best of all runs in `results/results.csv`; `*_pinned` rows copy from page-locked host memory. Every GPU count equals the serial CPU count. Charts: `results/chart_*.png`.

<!-- RESULTS:START -->
### Wiki-Vote  (n=8298, m=100762)

Speedup columns use kernel-only time for GPU rows. OMP reference: cpu_omp2.

| impl | GPU | triangles | kernel/CPU ms | end-to-end ms (GPU) | speedup vs cpu_serial | speedup vs OMP | check |
|---|---|---|---|---|---|---|---|
| cpu_serial | - | 608389 | 35.235 | - | - | - | ok |
| cpu_omp2 | - | 608389 | 21.345 | - | 1.65x | - | ok |
| gpu_v1_edge_per_thread | Tesla T4 | 608389 | 0.134 | 1.105 | 262.95x | 159.29x | ok |
| gpu_v1_edge_per_thread_pinned | Tesla T4 | 608389 | 0.137 | 0.268 | 257.19x | 155.80x | ok |
| gpu_v2_warp_per_edge | Tesla T4 | 608389 | 0.174 | 1.145 | 202.50x | 122.67x | ok |
| gpu_v2_warp_per_edge_pinned | Tesla T4 | 608389 | 0.176 | 0.307 | 200.20x | 121.28x | ok |
| gpu_v3_block_smem | Tesla T4 | 608389 | 0.162 | 1.000 | 217.50x | 131.76x | ok |
| gpu_v3_block_smem_pinned | Tesla T4 | 608389 | 0.164 | 0.246 | 214.85x | 130.15x | ok |
| gpu_v4_task_warp | Tesla T4 | 608389 | 0.183 | 2.318 | 192.54x | 116.64x | ok |
| gpu_v4_task_warp_pinned | Tesla T4 | 608389 | 0.186 | 1.871 | 189.44x | 114.76x | ok |
| gpu_v5_g16_per_edge | Tesla T4 | 608389 | 0.128 | 1.099 | 275.27x | 166.76x | ok |
| gpu_v5_g16_per_edge_pinned | Tesla T4 | 608389 | 0.129 | 0.260 | 273.14x | 165.47x | ok |
| gpu_v5_g2_per_edge | Tesla T4 | 608389 | 0.141 | 1.112 | 249.89x | 151.38x | ok |
| gpu_v5_g2_per_edge_pinned | Tesla T4 | 608389 | 0.143 | 0.274 | 246.40x | 149.27x | ok |
| gpu_v5_g4_per_edge | Tesla T4 | 608389 | 0.111 | 1.081 | 317.43x | 192.30x | ok |
| gpu_v5_g4_per_edge_pinned | Tesla T4 | 608389 | 0.111 | 0.242 | 317.43x | 192.30x | ok |
| gpu_v5_g8_per_edge | Tesla T4 | 608389 | 0.109 | 1.080 | 323.26x | 195.83x | ok |
| gpu_v5_g8_per_edge_pinned | Tesla T4 | 608389 | 0.111 | 0.242 | 317.43x | 192.30x | ok |
| gpu_v6_vertex_per_thread | Tesla T4 | 608389 | 0.983 | 1.821 | 35.84x | 21.71x | ok |
| gpu_v6_vertex_per_thread_pinned | Tesla T4 | 608389 | 1.001 | 1.083 | 35.20x | 21.32x | ok |
| gpu_v7_edge_search_src | Tesla T4 | 608389 | 0.138 | 0.976 | 255.33x | 154.67x | ok |
| gpu_v7_edge_search_src_pinned | Tesla T4 | 608389 | 0.140 | 0.220 | 251.68x | 152.46x | ok |
| gpu_v8_warp_vertex_hash | Tesla T4 | 608389 | 0.219 | 1.057 | 160.89x | 97.47x | ok |
| gpu_v8_warp_vertex_hash_pinned | Tesla T4 | 608389 | 0.219 | 0.301 | 160.89x | 97.47x | ok |
| gpu_v9_edge_sorted | Tesla T4 | 608389 | 0.139 | 3.090 | 253.49x | 153.56x | ok |
| gpu_v9_edge_sorted_pinned | Tesla T4 | 608389 | 0.140 | 2.842 | 251.68x | 152.46x | ok |

### as-Skitter  (n=1696415, m=11095298)

Speedup columns use kernel-only time for GPU rows. OMP reference: cpu_omp2.

| impl | GPU | triangles | kernel/CPU ms | end-to-end ms (GPU) | speedup vs cpu_serial | speedup vs OMP | check |
|---|---|---|---|---|---|---|---|
| cpu_serial | - | 28769868 | 1641.447 | - | - | - | ok |
| cpu_omp2 | - | 28769868 | 1036.358 | - | 1.58x | - | ok |
| gpu_v1_edge_per_thread | Tesla T4 | 28769868 | 8.928 | 29.593 | 183.85x | 116.08x | ok |
| gpu_v1_edge_per_thread_pinned | Tesla T4 | 28769868 | 9.122 | 16.906 | 179.94x | 113.61x | ok |
| gpu_v2_warp_per_edge | Tesla T4 | 28769868 | 18.978 | 39.644 | 86.49x | 54.61x | ok |
| gpu_v2_warp_per_edge_pinned | Tesla T4 | 28769868 | 19.129 | 26.913 | 85.81x | 54.18x | ok |
| gpu_v3_block_smem | Tesla T4 | 28769868 | 22.560 | 33.792 | 72.76x | 45.94x | ok |
| gpu_v3_block_smem_pinned | Tesla T4 | 28769868 | 23.095 | 27.273 | 71.07x | 44.87x | ok |
| gpu_v4_task_warp | Tesla T4 | 28769868 | 20.523 | 231.285 | 79.98x | 50.50x | ok |
| gpu_v4_task_warp_pinned | Tesla T4 | 28769868 | 20.616 | 215.481 | 79.62x | 50.27x | ok |
| gpu_v5_g16_per_edge | Tesla T4 | 28769868 | 13.053 | 33.719 | 125.75x | 79.40x | ok |
| gpu_v5_g16_per_edge_pinned | Tesla T4 | 28769868 | 13.167 | 20.950 | 124.66x | 78.71x | ok |
| gpu_v5_g2_per_edge | Tesla T4 | 28769868 | 8.181 | 28.846 | 200.64x | 126.68x | ok |
| gpu_v5_g2_per_edge_pinned | Tesla T4 | 28769868 | 8.190 | 15.974 | 200.42x | 126.54x | ok |
| gpu_v5_g4_per_edge | Tesla T4 | 28769868 | 8.116 | 28.781 | 202.25x | 127.69x | ok |
| gpu_v5_g4_per_edge_pinned | Tesla T4 | 28769868 | 8.252 | 16.035 | 198.92x | 125.59x | ok |
| gpu_v5_g8_per_edge | Tesla T4 | 28769868 | 9.480 | 30.146 | 173.15x | 109.32x | ok |
| gpu_v5_g8_per_edge_pinned | Tesla T4 | 28769868 | 9.622 | 17.405 | 170.59x | 107.71x | ok |
| gpu_v6_vertex_per_thread | Tesla T4 | 28769868 | 16.446 | 27.677 | 99.81x | 63.02x | ok |
| gpu_v6_vertex_per_thread_pinned | Tesla T4 | 28769868 | 16.519 | 20.697 | 99.37x | 62.74x | ok |
| gpu_v7_edge_search_src | Tesla T4 | 28769868 | 8.851 | 20.082 | 185.45x | 117.09x | ok |
| gpu_v7_edge_search_src_pinned | Tesla T4 | 28769868 | 9.023 | 13.201 | 181.92x | 114.86x | ok |
| gpu_v8_warp_vertex_hash | Tesla T4 | 28769868 | 12.628 | 23.860 | 129.98x | 82.07x | ok |
| gpu_v8_warp_vertex_hash_pinned | Tesla T4 | 28769868 | 12.684 | 16.862 | 129.41x | 81.71x | ok |
| gpu_v9_edge_sorted | Tesla T4 | 28769868 | 9.101 | 250.450 | 180.36x | 113.87x | ok |
| gpu_v9_edge_sorted_pinned | Tesla T4 | 28769868 | 9.030 | 225.820 | 181.78x | 114.77x | ok |

### com-LiveJournal  (n=4036538, m=34681189)

Speedup columns use kernel-only time for GPU rows. OMP reference: cpu_omp2.

| impl | GPU | triangles | kernel/CPU ms | end-to-end ms (GPU) | speedup vs cpu_serial | speedup vs OMP | check |
|---|---|---|---|---|---|---|---|
| cpu_serial | - | 177820130 | 10742.980 | - | - | - | ok |
| cpu_omp2 | - | 177820130 | 6362.185 | - | 1.69x | - | ok |
| gpu_v1_edge_per_thread | Tesla T4 | 177820130 | 55.976 | 117.600 | 191.92x | 113.66x | ok |
| gpu_v1_edge_per_thread_pinned | Tesla T4 | 177820130 | 56.062 | 79.929 | 191.63x | 113.48x | ok |
| gpu_v2_warp_per_edge | Tesla T4 | 177820130 | 69.589 | 131.213 | 154.38x | 91.43x | ok |
| gpu_v2_warp_per_edge_pinned | Tesla T4 | 177820130 | 70.039 | 93.906 | 153.39x | 90.84x | ok |
| gpu_v3_block_smem | Tesla T4 | 177820130 | 73.255 | 105.854 | 146.65x | 86.85x | ok |
| gpu_v3_block_smem_pinned | Tesla T4 | 177820130 | 73.703 | 86.316 | 145.76x | 86.32x | ok |
| gpu_v4_task_warp | Tesla T4 | 177820130 | 75.950 | 952.928 | 141.45x | 83.77x | ok |
| gpu_v4_task_warp_pinned | Tesla T4 | 177820130 | 75.793 | 850.487 | 141.74x | 83.94x | ok |
| gpu_v5_g16_per_edge | Tesla T4 | 177820130 | 57.154 | 118.778 | 187.97x | 111.32x | ok |
| gpu_v5_g16_per_edge_pinned | Tesla T4 | 177820130 | 57.613 | 81.479 | 186.47x | 110.43x | ok |
| gpu_v5_g2_per_edge | Tesla T4 | 177820130 | 56.901 | 118.525 | 188.80x | 111.81x | ok |
| gpu_v5_g2_per_edge_pinned | Tesla T4 | 177820130 | 57.315 | 81.182 | 187.44x | 111.00x | ok |
| gpu_v5_g4_per_edge | Tesla T4 | 177820130 | 49.147 | 110.771 | 218.59x | 129.45x | ok |
| gpu_v5_g4_per_edge_pinned | Tesla T4 | 177820130 | 49.530 | 73.397 | 216.90x | 128.45x | ok |
| gpu_v5_g8_per_edge | Tesla T4 | 177820130 | 51.112 | 112.736 | 210.19x | 124.48x | ok |
| gpu_v5_g8_per_edge_pinned | Tesla T4 | 177820130 | 51.817 | 75.684 | 207.33x | 122.78x | ok |
| gpu_v6_vertex_per_thread | Tesla T4 | 177820130 | 115.479 | 148.078 | 93.03x | 55.09x | ok |
| gpu_v6_vertex_per_thread_pinned | Tesla T4 | 177820130 | 115.035 | 127.649 | 93.39x | 55.31x | ok |
| gpu_v7_edge_search_src | Tesla T4 | 177820130 | 57.330 | 89.929 | 187.39x | 110.97x | ok |
| gpu_v7_edge_search_src_pinned | Tesla T4 | 177820130 | 57.952 | 70.565 | 185.38x | 109.78x | ok |
| gpu_v8_warp_vertex_hash | Tesla T4 | 177820130 | 61.315 | 93.914 | 175.21x | 103.76x | ok |
| gpu_v8_warp_vertex_hash_pinned | Tesla T4 | 177820130 | 61.489 | 74.103 | 174.71x | 103.47x | ok |
| gpu_v9_edge_sorted | Tesla T4 | 177820130 | 61.092 | 979.055 | 175.85x | 104.14x | ok |
| gpu_v9_edge_sorted_pinned | Tesla T4 | 177820130 | 61.764 | 1085.405 | 173.94x | 103.01x | ok |

### email-Enron  (n=36692, m=183831)

Speedup columns use kernel-only time for GPU rows. OMP reference: cpu_omp2.

| impl | GPU | triangles | kernel/CPU ms | end-to-end ms (GPU) | speedup vs cpu_serial | speedup vs OMP | check |
|---|---|---|---|---|---|---|---|
| cpu_serial | - | 727044 | 33.435 | - | - | - | ok |
| cpu_omp2 | - | 727044 | 20.143 | - | 1.66x | - | ok |
| gpu_v1_edge_per_thread | Tesla T4 | 727044 | 0.158 | 0.754 | 211.61x | 127.49x | ok |
| gpu_v1_edge_per_thread_pinned | Tesla T4 | 727044 | 0.158 | 0.351 | 211.61x | 127.49x | ok |
| gpu_v2_warp_per_edge | Tesla T4 | 727044 | 0.252 | 0.847 | 132.68x | 79.93x | ok |
| gpu_v2_warp_per_edge_pinned | Tesla T4 | 727044 | 0.250 | 0.443 | 133.74x | 80.57x | ok |
| gpu_v3_block_smem | Tesla T4 | 727044 | 0.410 | 0.746 | 81.55x | 49.13x | ok |
| gpu_v3_block_smem_pinned | Tesla T4 | 727044 | 0.408 | 0.527 | 81.95x | 49.37x | ok |
| gpu_v4_task_warp | Tesla T4 | 727044 | 0.264 | 3.058 | 126.65x | 76.30x | ok |
| gpu_v4_task_warp_pinned | Tesla T4 | 727044 | 0.267 | 2.276 | 125.22x | 75.44x | ok |
| gpu_v5_g16_per_edge | Tesla T4 | 727044 | 0.159 | 0.754 | 210.28x | 126.69x | ok |
| gpu_v5_g16_per_edge_pinned | Tesla T4 | 727044 | 0.160 | 0.353 | 208.97x | 125.89x | ok |
| gpu_v5_g2_per_edge | Tesla T4 | 727044 | 0.126 | 0.721 | 265.36x | 159.87x | ok |
| gpu_v5_g2_per_edge_pinned | Tesla T4 | 727044 | 0.127 | 0.340 | 263.27x | 158.61x | ok |
| gpu_v5_g4_per_edge | Tesla T4 | 727044 | 0.104 | 0.700 | 321.49x | 193.68x | ok |
| gpu_v5_g4_per_edge_pinned | Tesla T4 | 727044 | 0.105 | 0.298 | 318.43x | 191.84x | ok |
| gpu_v5_g8_per_edge | Tesla T4 | 727044 | 0.117 | 0.712 | 285.77x | 172.16x | ok |
| gpu_v5_g8_per_edge_pinned | Tesla T4 | 727044 | 0.117 | 0.310 | 285.77x | 172.16x | ok |
| gpu_v6_vertex_per_thread | Tesla T4 | 727044 | 0.894 | 1.230 | 37.40x | 22.53x | ok |
| gpu_v6_vertex_per_thread_pinned | Tesla T4 | 727044 | 0.903 | 1.022 | 37.03x | 22.31x | ok |
| gpu_v7_edge_search_src | Tesla T4 | 727044 | 0.160 | 0.496 | 208.97x | 125.89x | ok |
| gpu_v7_edge_search_src_pinned | Tesla T4 | 727044 | 0.161 | 0.280 | 207.67x | 125.11x | ok |
| gpu_v8_warp_vertex_hash | Tesla T4 | 727044 | 0.203 | 0.539 | 164.70x | 99.23x | ok |
| gpu_v8_warp_vertex_hash_pinned | Tesla T4 | 727044 | 0.203 | 0.322 | 164.70x | 99.23x | ok |
| gpu_v9_edge_sorted | Tesla T4 | 727044 | 0.168 | 3.714 | 199.02x | 119.90x | ok |
| gpu_v9_edge_sorted_pinned | Tesla T4 | 727044 | 0.170 | 3.233 | 196.68x | 118.49x | ok |

GPU(s) in results: Tesla T4
<!-- RESULTS:END -->

## Headline (Tesla T4, burn-in on, every count verified against the serial CPU count)
| graph | CPU serial / OMP-2 | v1 kernel | best kernel (v5, G=4) | best end-to-end (v7, pinned) |
|---|---|---|---|---|
| as-Skitter (1.7M vertices, 11.1M edges, 28,769,868 triangles) | 1641 / 1036 ms | 8.93 ms (184x / 116x) | 8.12 ms (202x / 128x) | 13.20 ms (124x / 78x) |
| com-LiveJournal (4.0M vertices, 34.7M edges, 177,820,130 triangles) | 10,743 / 6,362 ms | 55.98 ms (192x / 114x) | 49.15 ms (219x / 129x) | 70.57 ms (152x / 90x) |

Speedups in brackets are vs CPU serial / vs CPU OMP-2. Kernel-only excludes copying the graph to the GPU; end-to-end includes it (pinned host memory). Both exclude reading the file and building the oriented CSR, which is the same for CPU and GPU. The CPU baseline is a plain merge counter on the 2 cores of the Colab VM, so these are speedups against a modest baseline, not a tuned CPU library. No cuGraph comparison could be obtained (see limitations). v5 (G=4) beats v1 by about 10-12 % on both large graphs in every run, a small but repeatable gain; differences between v1, v5 and v7 under about 10 % on a single run are noise.

## Kernels
- **v1 (edge per thread):** one thread per oriented edge (u,v), sequential merge of N+(u) and N+(v). Simple; neighbouring threads read unrelated lists.
- **v2 (warp per edge):** one warp per edge; lanes take elements of the smaller list and binary-search them in the larger. One atomic per warp.
- **v3 (block per vertex, shared memory):** N+(u) staged in shared memory (when it fits SM_CAP), warps take neighbours v, block reduction, one atomic per block.
- **v4 (task-split warps):** host splits each edge's work into fixed chunks, one warp per task, to balance hubs.
- **v5 (G threads per edge, G = 2/4/8/16):** between v1 (1 thread) and v2 (32 lanes); tests whether lane utilisation matters when lists are short.
- **v6 (vertex per thread):** no `src` array needed (half the upload).
- **v7 (edge per thread, source vertex found by binary search on `rowptr`):** v1 without the `src` array; computes instead of transferring.
- **v8 (`tc_warp_vertex_hash`, warp per vertex, shared-memory hash set):** one warp per vertex u; N+(u) is inserted (atomicCAS, linear probing) into a private 512-slot shared-memory hash set (`HASH_SLOTS`), then all lanes probe it with the elements of every N+(v). Falls back to binary search in global memory if out-degree > `HASH_SLOTS/2`. Measured: slower than v1 (see below).
- **v9 (`tc_edge_sorted`, edges in work-sorted order):** v1's merge, but thread i handles edge `perm[i]`; `perm` is built on the host by a counting sort on floor(log2(min(du,dv)+1)) (`build_edge_order` in `graph.h`) so a warp's lanes get similar work. Measured: no kernel speedup, large host cost (see below).

## What worked, what did not (measured on the T4, see docs/analysis.md)
- **A measurement bug found and fixed:** without a GPU warm-up, v1 (always timed first) ran at a lower SM clock and looked 2x slower (14.6 ms) than v5/v7 (8 ms). Adding a 1.5 s burn-in before timing moved v1 to 8.4-9.4 ms (best of 10, repeated runs) while v5 and v7 stayed at 7.3-9.5 ms. Nsight Compute had already shown v1, v5(G=2) and v7 need about the same number of cycles (10.1 M, 10.2 M, 10.8 M at a fixed clock). Lesson: warm the GPU before timing, and check timing differences against a cycle-level profile.
- **Result:** v1, v5 (G=2/4) and v7 are within roughly 10 % of each other (about 8-9 ms on as-Skitter); simple edge-per-thread is already close to the best of what was tried. The structured variants lose: v2 warp-per-edge 19 ms, v3 shared memory 23 ms, v4 task-split 21 ms (plus 175-250 ms host task build), v6 vertex-per-thread 16 ms.
- **Why v2 loses (profiler):** it executes 1.50 G warp instructions vs 0.57 G for v1 (2.6x) and needs 21.5 M vs 10.1 M cycles at a fixed clock. After degree orientation lists are short (max out-degree 74 / 70 / 231), so 32 lanes per edge mostly add overhead.
- **Why v1 is not faster (profiler):** L1/TEX pipeline about 90 % utilised, DRAM 10 %, 11.1 of 32 lanes active per warp. The kernels are limited by cache-pipeline pressure and divergence, not DRAM bandwidth, so the 1-5 % effective-bandwidth estimate does not mean bandwidth is wasted.
- **Transfers:** pinned host memory cut the as-Skitter copy from about 21 ms to 8-10 ms (95.5 MB); v3/v6/v7 upload about half as much because they skip `src`. Copy time is as large as kernel time at this graph size.
- **v8 hash set (tried, did not help):** 12.6 ms vs 8.9 ms for v1 on as-Skitter (61.3 vs 56.0 ms on com-LiveJournal). Nsight Compute (fixed clock): 806 M warp instructions vs 573 M for v1 (+41 %), 15.2 M cycles vs 10.1 M (+51 %), achieved occupancy 59 % (16 KB static shared memory per block), L1/TEX 50 %, DRAM 5 %. So it does more instruction work and runs at lower occupancy; my untested hypothesis is that clearing 512 slots for every vertex dominates, since average out-degree is only about 6.5.
- **v9 work-sorted edges (tried, did not help):** 9.1 ms kernel (same as v1) but the host sort takes about 205 ms, so end-to-end is 225 ms. Nsight Compute: 471 M warp instructions vs 573 M for v1 (-18 %, less divergence) yet 10.39 M cycles vs 10.08 M (no faster), L1/TEX still 88 %, DRAM up to 17 %. Lesson: reducing divergence does not help when the L1/TEX pipeline is the limiter.
- **Large graph:** on com-LiveJournal (34.7M edges, max out-degree 524) the ranking is the same as on the smaller graphs: v5 (G=4) 49.1 ms, v8 61.3, v1 56.0, v7 57.3, v2 69.6, v3 73.3, v4 76.0, v6 115.5 ms. Pinned memory cut its copy from 61.6 ms to 23.9 ms.
- **Tried, did not help (earlier):** v3, v4 (hub imbalance is not the problem on these graphs), v6 (only n threads: 6-8x slower on the small graphs), v2.

## GPU densest subgraph pipeline (stage 5) -- measured on a T4

**STATUS: compiled with nvcc (`-arch=sm_75`) and run on a Tesla T4 (free Colab) on 2026-10-11 on Wiki-Vote, Email-Enron, as-Skitter and com-LiveJournal. On all four the GPU result was bit-exact against `cpu_batch_peel` (the program exits 1 and prints no rows otherwise).** The numbers below come from the committed Colab logs `results/colab_run_stage6_small_T4_2026-10-11.log` and `results/colab_run_stage6_large_T4_2026-10-11.log` (single runs of `bin/gpu_densest <graph> 1 10 --pinned`, eps = 1/10, burn-in on, kernel time = best of the program's repeats); they have not been through `run_bench.sh`, so `results/results_densest.csv` does not exist yet.

| graph | n / m | GPU kernel-only | GPU end-to-end (+ h2d) | serial CPU batch peel | 2-core OpenMP batch peel | GPU density / sequential greedy |
|---|---|---|---|---|---|---|
| Wiki-Vote | 8.3k / 101k | 0.709 ms | 0.895 ms | 72.8 ms | 81.8 ms | 0.854 |
| Email-Enron | 36.7k / 184k | 0.862 ms | 1.059 ms | 59.0 ms | 56.7 ms | 0.861 |
| as-Skitter | 1.70M / 11.1M | 33.1 ms | 41.0 ms | 3,726 ms | 3,057 ms | 0.637 |
| com-LiveJournal | 4.04M / 34.7M | 333.6 ms | 357.5 ms | 17,640 ms | 21,370 ms | not run (`--no-greedy`) |

Speed-ups are for the SAME algorithm on CPU and GPU (kernel-only vs serial CPU: about 100x, 68x, 112x, 53x). The result is an approximation (bound 3(1+eps)); on these graphs the batch result has 64-86% of the density of sequential greedy, so it is a fast, coarser answer, not a replacement for greedy or the exact solver. Caveats: CPU numbers come from the 2-core Colab VM (OpenMP is not faster than serial on two of the four graphs); one run per graph; the sequential greedy ratio on as-Skitter uses the greedy density printed by `gpu_dsg` (3366.622328).

**Problem.** Find a vertex set `S` maximising `density(S) = triangles(S) / |S|`, where `triangles(S)` counts the triangles whose three corners are all in `S` (the triangle, i.e. k=3 clique, densest subgraph).

**Algorithm (GPU batch peeling).** Repeat until no vertex is alive:
1. count `T`, the triangles of the subgraph induced by the alive vertices, and `tri[v]`, the number of those triangles containing `v` (`tc_per_vertex_alive`: the v1 edge-per-thread merge on the oriented CSR, skipping dead endpoints and dead third vertices, three `atomicAdd`s per triangle);
2. remember the density `T/nalive` if it beats the best so far (compared as integer fractions by cross multiplication, no floating point);
3. remove every alive vertex with `tri[v] <= (1+eps) * 3T/nalive`, i.e. at most `(1+eps)` times the average participation (`mark_peel`, computed as `tri[v]*nalive*den <= 3*(den+num)*T` in uint64, `eps = num/den`, default 1/10).

`removed_round[v]` records the round in which `v` left; the best subgraph is `{v : removed_round[v] >= best_round}`. Because `sum_v tri[v] = 3T`, the minimum is at most the average, so every round removes at least one vertex, and by Markov's inequality at least a fraction `eps/(1+eps)` of the alive vertices, giving `O(log n / eps)` rounds.

**Approximation guarantee, with caveat.** The literature (Bahmani-Kumar-Vassilvitskii for edges; the k-clique generalisation is attributed to Tsourakakis, 2015 - cited from memory, check the paper) gives: the best density seen is within a factor `3(1+eps)` of the optimum. Sketch: every vertex of an optimal set `S*` has participation at least `rho*` inside `S*`; in the first round that removes a vertex of `S*`, the alive set contains `S*`, so that vertex has participation at least `rho*`, yet was removed because its participation was at most `3(1+eps)` times the current density. This is a worst-case bound only and says nothing about how close the result is on a real graph. Here it is merely checked by brute force on tiny random graphs (n <= 16) in `make test`. The quality baseline is `cpu_greedy_peel`, the classical sequential peeling (remove the single vertex with the fewest triangles; a 3-approximation).

**Design decisions to be able to explain.**
- The graph is never compacted: dead vertices are flagged in a byte array and the CSR stays as uploaded. Late rounds therefore still launch one thread per original edge (most exit after reading two flags). Compacting is an obvious later optimisation and is not claimed to help.
- One host synchronisation per round: `mark_peel` reads `T` from device memory, then `{T, removed}` (16 bytes) are copied back once; best-density tracking and the alive count live on the host. Simple on purpose; cost is (rounds x sync latency), not measured.
- Atomics: `tri[]` receives three `atomicAdd`s per triangle, so hub vertices become contention points. Alternative ideas (per-edge local counts, warp-aggregated atomics) are described in comments in `src/kernels.cuh` and are NOT implemented.
- Arithmetic: tri up to ~1e8, nalive up to ~4e6, den small gives products of ~4e15 < 1.8e19; `peel_arith_ok()` checks the real numbers (T < 2^32 for the uint32 counters, no uint64 overflow in either side of the test) and the program refuses to run otherwise.
- The GPU result must equal `cpu_batch_peel` exactly (best triangles, best vertices, best round, number of rounds and the full `removed_round` array); on any mismatch `gpu_densest` exits 1 and prints no rows.

**Build and run.**

    make cpu                                   # builds bin/cpu_tc and bin/cpu_densest (no GPU needed)
    make test                                  # CPU emulator tests incl. brute-force densest subgraph (no GPU needed)
    make ARCH=sm_75                            # on Colab: also builds bin/gpu_tc and bin/gpu_densest
    ./bin/gpu_densest data/Wiki-Vote.txt                      # eps = 1/10, default
    ./bin/gpu_densest data/email-Enron.txt 1 4 --verbose      # eps = 1/4, per-round diagnostics on stderr
    ./bin/gpu_densest data/as-Skitter.txt 1 10 --no-greedy    # skip the slow sequential greedy on big graphs
    bash scripts/run_bench.sh --densest data/Wiki-Vote.txt    # appends to results/results_densest.csv
    python3 scripts/make_densest_table.py --write-readme      # fills the markers below from the CSV

Options: `--pinned`, `--no-burnin` (burn-in of about 1.5 s is on by default), `--repeats N`, `--no-greedy`, `--verbose`. CSV columns: `impl,graph,n,m,best_triangles,best_vertices,density,rounds,time_ms,h2d_ms,total_ms`; implementations `cpu_batch_peel`, `cpu_batch_peel_omp` (when built with OpenMP), `cpu_greedy_peel`, `gpu_batch_peel`. `time_ms` of the GPU row = all peeling rounds including the per-round host syncs, excluding uploads (`h2d_ms`), burn-in, the final `removed_round` download and verification; best of N. The `density` column is a 6-decimal display value; comparisons use the integers. The density ratio gpu/greedy is printed on stderr.

<!-- DENSEST:START -->
Measured numbers for this stage are in the table above (from the committed Colab logs). This block is reserved for `scripts/make_densest_table.py --write-readme`, which fills it from `results/results_densest.csv` once `run_bench.sh --densest` has been run.
<!-- DENSEST:END -->

**Tried, did not help.** Nothing was tried for this stage beyond the first version. Not done and therefore not claimed: compacting the edge list in late rounds, warp-aggregated atomics. Not profiled with Nsight Compute either, so no explanation of where the 33 ms on as-Skitter goes is offered.

## Stage 6: paper algorithms (Greedy++, triangle core decomposition, CoreExact) with GPU core decomposition -- first version measured on a T4, stages 6b and 6c NOT YET MEASURED

**STATUS (2026-10-11).** The first version of this stage (commit `e0a620c`, level-synchronous core decomposition that RECOUNTS all triangles every round) compiled with nvcc and ran on a Tesla T4 on Wiki-Vote, Email-Enron and as-Skitter; its GPU core numbers matched CPU Algorithm 3 element by element on all three, and `--verify-full` on Wiki-Vote gave an identical optimum with and without pruning. On com-LiveJournal that version did not finish within the 600 s limit I had set (`exit 124`). Logs: `results/colab_run_stage6_small_T4_2026-10-11.log`, `results/colab_run_stage6_large_T4_2026-10-11.log`. **Two later changes are written and emulator-tested only, NOT compiled with nvcc and NOT measured: the incremental frontier core decomposition (6c) and the Greedy++-seeded CoreExact (6b), both described below.** There is no `results/results_dsg.csv` yet. (The `.cu` files were syntax-checked with clang host-only against a stub header, which is not a real nvcc compile.)

**Measured on the T4 (recount version, `e0a620c`; GPU numbers are the program's best-of-N kernel time including the per-round sync, excluding uploads):**

| graph | rounds | kmax | GPU core decomposition (recount) | serial CPU Alg 3 | pruned graph after greedy lower bound |
|---|---|---|---|---|---|
| Wiki-Vote | 819 | 600 | 136.0 ms (134.7 in a second run) | 161.1 ms (116.1 in a second run) | 473 of 8,298 vertices |
| Email-Enron | 1,105 | 509 | 196.5 ms | 158.1 ms | 390 of 36,692 |
| as-Skitter | 5,626 | 4,549 | 27,209.5 ms | 13,347.4 ms | 421 of 1,696,415 |
| com-LiveJournal | - | - | timed out (> 600 s) | not reached | - |

Findings, stated plainly:
- **The recount design is slower than the serial CPU algorithm at scale** (about 2x slower on as-Skitter, 1.24x slower on Enron, no better on Wiki-Vote). Time per round is about 0.17 ms on Wiki-Vote and Enron and 4.8 ms on as-Skitter, i.e. the cost tracks rounds x (full triangle recount). I did not profile it with Nsight Compute, so that explanation follows from the design and the round counts, not from a profile.
- **The exact stage dominates the pipeline.** On Wiki-Vote, CoreExact on the pruned graph took 12.06 s (27 max-flow calls) of a 12.31 s pipeline; the GPU core decomposition was about 1% of it. CoreExact on the pruned graph (12.1 s) was barely faster than on the full graph (12.9 s) because Alg 4 already prunes by itself. So this version does NOT make exact densest-subgraph search faster, and I do not claim that.
- **The core-based pruning itself is effective:** the sequential-greedy density as a lower bound plus the `ceil(lb)`-core leaves 421 of 1.7M vertices (as-Skitter). Greedy++ (T = 10) on the pruned graph returned the exact optimum on Wiki-Vote (235,020 triangles / 466 vertices = 504.334764, equal to the `--verify-full` result); on as-Skitter it equalled the greedy value 3366.622328 and on Enron it gave 404.456186 vs greedy 404.451948 (no exact value for those two graphs).

**Stage 6b (written, emulator-tested only, not measured): Greedy++-seeded CoreExact.** Because the exact stage dominates, `core_exact` now accepts an optional seed (a real subgraph, e.g. the Greedy++ result; its set and triangle count are re-verified, a bogus seed is rejected). It raises the lower bound `l` and starts the binary search just above `l` (`ceil(l*Q)/Q`): if the first max-flow finds nothing denser, `l` is proven optimal in one flow call (two distinct densities differ by at least `2/Q` on the grid, so nothing lies in `(l, ceil(l*Q)/Q]`). `gpu_dsg` now runs Greedy++ first and then the seeded CoreExact (row `cpu_core_exact_pruned_seeded`). Emulator tests: seeded results equal brute force / Exact on all test graphs, bogus seeds are rejected, 2 deliberate mutants are caught. The effect on running time is unknown until it is run on a real graph (on the 484 tiny test graphs the number of max-flow calls dropped from 439 to 346, which is not evidence for large graphs).

**Stage 6c (written, emulator-tested only, not measured): incremental frontier core decomposition.** Replaces the full recount per round: `tri[]` is computed once, and each round only subtracts the triangles that contain a removed vertex (`core_find_frontier`, `core_update_frontier` = one warp per frontier vertex over the undirected sorted CSR, `core_retire_frontier`; a triangle is handled by its smallest-id frontier vertex so it is subtracted exactly once). The default `gpu_core_decomp` row of `gpu_dsg` is now this version; `--with-recount` also runs the old one (`gpu_core_decomp_recount`) for the comparison. Same exactness gate: element-wise equality with CPU Alg 3 or no timing. Emulator tests: identical core numbers and round counts versus the recount loop on all test files, tiny graphs and 120 hub graphs (vertices with 40-100 neighbours, which the lane-striding of the update kernel needs); mutants that drop the owner rule, ignore dead vertices or break the lane stride are caught (two further mutants survived and are equivalent: skipping the binary-search start, flipping the owner order). Risks: a hub in the frontier is handled by a single warp (load imbalance), the undirected CSR doubles the edge memory (about 280 MB extra on com-LiveJournal), and whether any of this is actually faster than the serial CPU algorithm is not known until it runs.

**Source papers (taken from the PDFs, algorithm numbers are cited in the code comments).** Fang, Yu, Cheng, Lakshmanan, Lin, *Efficient Algorithms for Densest Subgraph Discovery* (PVLDB 2019): Alg 1 Exact (flow network + binary search), Alg 3 (k,Psi)-core decomposition, Theorem 1 (`k/|V_Psi| <= rho(R_k) <= kmax`), Lemma 5 (`rho_opt <= kmax`), Lemma 7 (the densest subgraph lies in the `ceil(rho_opt)`-core), Alg 4 CoreExact. Boob et al., *Flowless* (WWW 2020): Alg 1 Greedy (Charikar), Alg 2 Greedy++. Here Psi = triangle, density `rho(S) = triangles(S)/|S|`; Greedy++ treats every triangle as a hyperedge (`deg_H(u)` becomes the number of triangles of H containing u).

**What is implemented (all in `src/dsg.h`, CPU unless stated).**
| piece | paper | where it runs |
|---|---|---|
| per-vertex triangle degrees, `induced_subgraph` (valid oriented CSR through the shared `build_graph_from_edges`, which `load_graph` now also uses; Graph checksums on all data files identical before/after the refactor) | - | CPU |
| (k,Psi)-core decomposition `core_decomposition` | DSD Alg 3 | CPU (serial, lazy heap) **and GPU** (level-synchronous, below) |
| `greedy`, `greedy_pp(T)` (loads, remove argmin `l_u + tri_H(u)`, `l_u += tri_H(u)`; best density over all prefixes of all passes) | Flowless Alg 1, 2 | CPU only |
| Dinic max-flow, `exact_densest` | DSD Alg 1 | CPU only |
| `core_exact(G, core[])` (takes the core numbers as an input, so CPU or GPU numbers work) | DSD Alg 4 | CPU only |
| `prune_to_core` (keep `core >= ceil(lb)`) | DSD Lemma 7 | CPU |

The max-flow and Greedy++ stay on the CPU; only the core decomposition is on the GPU. The first measured run showed that the exact solve on the pruned graph dominates the pipeline (see above), so the GPU part is not claimed to make the exact pipeline faster.

**Exact arithmetic.** Densities are fractions `(triangles, vertices)` compared by cross multiplication in `unsigned __int128`; no floating point in any decision (`density_value` is display only). Flow: `alpha = a/Q` with `Q = 2 n (n-1)`, all capacities scaled by `Q` to integers (`s->v: Q*tri(v)`, `v->t: 3a`, `v->psi: Q`, `psi->v: infinity`), search until `hi - lo < 2` grid units = `u - l < 1/(n(n-1))`. Two distinct densities with denominators `<= n` differ by at least `1/(n(n-1))`, so the last set found is optimal. `build()` refuses networks whose capacities could reach `2^59`.

**GPU core decomposition.** Loop (`core_decomp_loop` in `dsg.h`, kernels in `src/kernels.cuh`): recount `tri[v]` on the alive subgraph with the existing `tc_per_vertex_alive`; `core_min_alive_tri` (warp-shuffle min, one `atomicMin` per warp) gives `m = min tri`; `core_peel_level` sets `k = max(k, m)` and removes every alive vertex with `tri[v] <= k`, writing `core[v] = k`. One 24-byte device-to-host copy per round (the host sync). The proof that this equals Alg 3 is a comment above `core_decomp_loop` (invariants: all alive vertices have core `>= k`, the `(k+1)`-core is inside the alive set, so a vertex with `tri <= k` has core exactly `k`). The two reductions are separate kernels because the minimum must be known to all threads before anybody decides (no grid-wide barrier inside a launch).

**Pipeline (`bin/gpu_dsg`).** (1) GPU core numbers, verified element-wise against CPU Alg 3 (mismatch: `MISMATCH`, exit 1, no timing printed). (2) `lb = max(density of kmax-core, greedy density)`; both are real subgraphs so `lb <= rho_opt`, and the densest subgraph lies in the `ceil(rho_opt)`-core hence in the `ceil(lb)`-core: drop all other vertices. The core numbers of the pruned graph equal the original ones for the surviving vertices (argument in `prune_to_core`; also asserted at run time against a fresh CPU decomposition of the pruned graph). (3) CPU CoreExact with those core numbers and CPU Greedy++ on the pruned graph. (4) `--verify-full`: CoreExact on the full graph with the CPU core numbers must give exactly the same optimum density, else `MISMATCH`, exit 1.

**Where the code deviates from / interprets the papers (decisions to be able to explain).**
- Alg 1's network has one node per `(h-1)`-clique, i.e. per **edge** for triangles (not one node per triangle): `v -> psi` has capacity 1 when `psi + v` is a triangle. Cut value = `3 (T - T(S)) + 3 alpha |S|`, derivation in the comment above `TriNet`.
- Alg 4 line 3 sets `u <- kmax` once; here `u` is reset to `kmax` for every component (the bound was only proved for the components already processed), `l` is not. `l` is kept as the exact density of the best subgraph found so far (>= the paper's alpha). The subgraph found by the initial check (lines 7-9) is recorded as a candidate. Pruning1 uses the residual graphs of the peeling, reconstructed from the core numbers alone as suffixes of the `(core, id)` order (every core is such a suffix), so the function needs no peeling order as input.
- Alg 4 line 17 ("remove some vertices from C") is implemented as: rebuild the network on `C ∩ ceil(alpha)-core` with the same integer grid, using the input core numbers (for `k >= k''` the k-core of a component is `{v in C : core[v] >= k}`).
- Alg 3 uses a lazy-deletion heap instead of bin sort (O(T log n) instead of linear in the number of triangle incidences).
- The text of the PDFs was extracted with `pdftotext` and is partly garbled in two-column pages; Alg 4 lines 6, 17 and 19 were reconstructed from the surrounding prose and the example (Example 5). If the original differs there, the exact result is unaffected (the pruning rules are justified independently in the comments, and the tests compare against brute force) but the amount of pruning could differ.

**How exactness is verified (so far: CPU / CPU-emulator only - nothing of this has run on a GPU).** `make test` runs `emu/emu_dsg.cpp`: (a) `--tiny`: 484 small graphs (n <= 12: zero-triangle graphs, one triangle, cliques, disconnected components, ties, graphs where plain greedy is suboptimal) against brute force over all `2^n` subsets - exact optimum, definition-based core numbers (max over subsets containing v of the minimum triangle degree), witnesses re-counted; `exact_densest`, `core_exact` (also through the prune pipeline), `core_decomposition` must match; greedy never exceeds and reaches 1/3 of the optimum, `greedy_pp` is monotone in T, never exceeds the optimum, and standalone `greedy_pp(T)` equals pass T of a 30-pass run; `induced_subgraph` output is validated structurally (orientation, sorted rows, `src`, `max_out_degree`, equality with a fresh `build_graph_from_edges`); the GPU kernels (`tc_per_vertex_alive`, `core_min_alive_tri`, `core_peel_level`) run through `emu/cuda_shim.h` inside the real `core_decomp_loop` and must give core numbers identical to Alg 3 (kernels also unit-tested on random alive masks). (b) `--mid`: 160 medium graphs with planted communities and 15 targeted graphs (several components searched / line-17 shrink) where Exact (Alg 1) is the oracle for CoreExact. (c) the four `data/t_*.txt` graphs. `python3 scripts/mutation_check.py` applies deliberate bugs (flip `>` in Alg 3, drop a decrement, `<=` -> `<` in the peel kernel, wrong flow capacities, no reverse capacity in Dinic, over-aggressive pruning, ...) to a temp copy and checks that `emu_dsg --tiny` fails. Greedy++ reached the brute-force optimum on all small test graphs within 30 passes but needed more than one pass on 14 of the graphs; on `data/t_er800.txt` it did **not** reach the optimum within 10 or 20 passes (1.008772 vs optimum 1.010714), so "Greedy++ converges to the optimum" is not something the tests assume.

**Known limitations / risks.**
- (Recount version, measured to be too slow, see above; superseded by 6c, which is unmeasured.) Every round recounts all triangles of the alive subgraph (one thread per ORIGINAL edge, exactly like stage 5). The number of rounds is `sum over levels of the peeling cascade depth`; unlike the (1+eps) batch peeling of stage 5 (`O(log n / eps)` rounds) it has no logarithmic bound - a chain-like triangle structure (e.g. a strip of triangles) needs about one round per vertex. On real graphs the round count is unknown until measured (`--verbose` prints it per round). The incremental update (stage 6c) is the remedy; it is written but not yet run on a GPU.
- `tri[]` is uint32, so the graph must have fewer than `2^32` triangles (checked on the host). The CPU greedy on the full graph can be slow on huge hubs (`--no-greedy` skips it; `lb` is then just the kmax-core density).
- `dsg.h` uses `unsigned __int128` in host code; it should be fine for nvcc's host compiler, but that is untested.
- CPU timings printed by `cpu_dsg` in this development container are not results and are not in this repository.

**Build and run (Colab).**

    make cpu && make test                       # CPU programs + all emulator tests (no GPU)
    make ARCH=sm_75                             # on Colab: bin/gpu_dsg as well
    ./bin/cpu_dsg data/Wiki-Vote.txt --T 10 --no-full-exact
    ./bin/gpu_dsg data/Wiki-Vote.txt --verify-full --verbose
    ./bin/gpu_dsg data/email-Enron.txt --pinned --T 10 --verify-full
    ./bin/gpu_dsg data/as-Skitter.txt --no-greedy --no-exact   # core decomposition only
    DSG_ARGS="--verify-full" bash scripts/run_bench.sh --dsg data/Wiki-Vote.txt   # appends to results/results_dsg.csv
    python3 scripts/mutation_check.py           # optional: show that the tests catch deliberate bugs

Options of `gpu_dsg`: `--with-recount`, `--T N`, `--repeats N`, `--pinned`, `--no-burnin`, `--no-exact`, `--no-greedy`, `--no-greedypp`, `--verify-full`, `--verbose`. CSV columns `impl,graph,n,m,triangles,vertices,density,time_ms,h2d_ms,total_ms,rounds,kmax`; rows `cpu_core_decomp`, `gpu_core_decomp[_pinned]` (kernel loop including the per-round sync, excluding uploads, burn-in, verification), `gpu_core_decomp_recount[_pinned]` (only with `--with-recount`), `cpu_greedy`, `prune_kcore`, `cpu_greedy_pp_pruned`, `cpu_core_exact_pruned_seeded`, `pipeline_gpu_core[_pinned]`, `pipeline_cpu_core`, `cpu_core_exact_full`; the exact column meanings are documented at the top of `src/gpu_dsg.cu`. GPU model is recorded by `run_bench.sh`; kernel-only and end-to-end (with host-to-device) times are separate columns, and the serial CPU core decomposition is in the same file for comparison. No OpenMP CPU variant of Alg 3 exists (it is a sequential min-first peeling), so a "vs OpenMP" comparison for the decomposition is not available; the README must say so when numbers exist.

**Tried, did not help.**
- Recounting every triangle on every round for the GPU (k,Psi)-core decomposition: 1.24x slower than the serial CPU algorithm on Email-Enron, 2.0x slower on as-Skitter (27.2 s vs 13.3 s), did not finish on com-LiveJournal within 600 s. Reason (from the design and the round counts, not from a profile): cost per round is a full triangle count, and the peeling needs thousands of rounds. Replaced by the incremental version (6c, unmeasured).
- Using the GPU to speed up the exact pipeline on Wiki-Vote: the GPU part is about 1% of the time because CoreExact's max-flow dominates (12.06 s of 12.31 s). Addressed on the algorithm side by seeding (6b, unmeasured), not by the GPU.
- Pruning before CoreExact: gave no real speed-up on Wiki-Vote (12.1 s pruned vs 12.9 s full), because CoreExact already prunes internally.

## Timing definitions
- `time_ms` for GPU rows = best of N warm launches of the kernel only (one untimed warm-up first).
- `h2d_ms` = host-to-device copy of exactly the arrays that kernel reads (v3, v6, v7, v8 do not read `src`; v4 also copies its task list, v9 also copies its edge order).
- `gpu_total_ms` = kernel + h2d (+ host time to build the task list for v4). File loading and CSR construction are common to every variant and excluded.

## Honest limitations
- Free shared Colab T4; clocks and neighbours vary. Same kernel on the same graph varied noticeably between runs (e.g. v2 on as-Skitter 19.1-27.9 ms; Wiki-Vote v1 0.26 ms in one run, 0.13 ms in another), so sub-millisecond results and gaps under about 20 % should not be over-read.
- Four graphs (largest 34.7M edges), one GPU model, a hand-written intersection only. A cuGraph comparison was attempted and failed (`pip install` of the RAPIDS cu12 packages on a CUDA 13 Colab image left cudf unusable: `libcudf.so` not found), so there is no library reference. The profile of v8 and v9 is from as-Skitter only.
- The T4 is a 2018 Turing GPU, not a current architecture; numbers do not transfer directly.
- Every GPU count is verified against the serial CPU count; a mismatching run is not reported.

## Stages
1. [x] Baseline: CSR loader, CPU baselines, v1, correctness check
2. [x] Kernels v2-v7, CPU emulator tests, T4 benchmark with pageable and pinned transfers
3. [x] Profile v1 vs v2 with Nsight Compute; bandwidth estimate; charts
4. [x] Full re-run (all graphs, pageable + pinned) with burn-in and regenerate tables/charts (done: results/colab_run_session3_burnin_full_T4.log); larger graph (e.g. com-LiveJournal)
5. [ ] Final write-up (docs/analysis.md), optional: try smaller `HASH_SLOTS` for v8, cuGraph reference with the preinstalled Colab RAPIDS (no pip install)
6. [ ] GPU densest-subgraph pipeline (stage 5 branch): code and emulator tests written; compile with nvcc, verify, benchmark and profile on Colab still to do
7. [ ] Paper algorithms (stage 6 branch): code and CPU/emulator tests written; compile with nvcc, verify, benchmark on Colab still to do
