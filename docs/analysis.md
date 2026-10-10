# Analysis (fill in AFTER measuring)

Every number here must come from `results/results.csv` or a profiler report you ran. Leave `TBD` until then.

## Problem
<!-- What is triangle counting, why does it matter (clustering coefficient, truss, densest subgraph), input sizes of the graphs used. -->
TBD

## Method
- Orientation trick: <!-- explain orienting each edge from lower (degree, id) to higher -->
- Why each triangle is counted exactly once: <!-- the unique lowest vertex u, edge (u,v), common out-neighbour w -->
- Why order by degree rather than id (effect on max out-degree): <!-- your explanation -->
- Intersection method: <!-- merge vs binary search, and why -->

## Kernel designs and why
| Version | Mapping | Why this design | Expected weakness (to be checked, not assumed) |
|---|---|---|---|
| v1 edge per thread | | | |
| v2 warp per edge | | | |
| v3 block per vertex + shared memory | | | |
| v4 task-split warps | | | |

## Results
GPU used: TBD (run on Colab)

| graph | cpu_serial ms | cpu_omp ms | v1 ms | v2 ms | v3 ms | v4 ms |
|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Kernel-only vs end-to-end (incl. host-to-device): TBD
Count verified against cpu_serial for all rows: TBD (yes/no)

## Bottleneck analysis
- What evidence shows the bottleneck? (profiler metric, or a timing comparison; cite the file/run) TBD
- What did you try to fix it? TBD
- What did not help, and your best explanation why? TBD
- Before/after numbers for the fix that worked: TBD (run on Colab)

## Limitations
<!-- Shared free GPU, one GPU model, noise, graphs tested, hand-written intersection only, no comparison against cuGraph/other libraries, tools that were blocked, etc. -->
TBD

## Interview questions I should be able to answer
Write your own answers in your own words once you understand them.
1. What is a warp, and what happens when lanes in a warp take different branches?
2. Why did one-thread-per-edge (v1) risk divergence on skewed graphs?
3. What is memory coalescing, and which kernel's loads are more or less coalesced? How would you check?
4. Why is binary search a reasonable intersection method on a GPU compared with a sequential merge?
5. What does shared memory buy in v3, what limits its use (SM_CAP), and what is a bank conflict?
6. What is occupancy, what limits it in each kernel, and why can higher occupancy still not mean faster?
7. Why do atomics cost time, and why do v2/v3 reduce them to one per warp / per block? How does a warp or block reduction work?
8. What is load imbalance with hub vertices, and how does the v4 task-splitting address it? What does it cost (host-side preprocessing)?
9. Why orient edges by degree, and what does it do to the worst-case work per edge?
10. How do you know the GPU result is correct? What would you test beyond equality on the SNAP graphs (e.g. the brute-force synthetic graph)?
11. Why report kernel-only and end-to-end time separately? When would the GPU not be worth it?
12. Is triangle counting compute-bound or memory-bound, and how would you tell from your measurements?
13. What would you try next if you had more time?


## Hypotheses to test (NOT results)
- Real graphs like as-Skitter have small average out-degree after degree orientation, so the smaller list is often < 32 elements and most lanes of a warp idle in v2/v3/v4. If so, warp-per-edge may not beat v1, and low warp efficiency would come from lane utilisation and random gathers, not necessarily hub serialisation. Check warp execution efficiency / `smsp__thread_inst_executed_per_inst_executed` before blaming load imbalance.
- v3 launches a full block per vertex; for low-degree vertices that is mostly barrier overhead. Compare against a warp-per-vertex variant.
- v1's single atomic per thread may add contention; test a version with a warp reduction.
- Each of these is a guess until the measurements say otherwise; keep the ones that turn out wrong in the write-up.


## Findings (Colab Tesla T4, 2026-10-10; numbers in results/results.csv, raw logs in results/colab_run_2026-10-10_T4*.log and results/ncu_as-Skitter_v1_v2_2026-10-10_T4.log)
All kernels v1-v7 (pageable and pinned) produced counts identical to the serial CPU count on Wiki-Vote, email-Enron and as-Skitter.

### Kernel comparison on as-Skitter (kernel-only ms, pageable run / pinned run)
v1 16.6 / 17.3, v2 24.8 / 19.1, v3 22.7 / 23.7, v4 20.3 / 20.7, v5 G=2 8.35 / 8.37, G=4 8.6 / 8.4, G=8 9.8 / 9.9, G=16 13.3 / 13.5, v6 16.5 / 16.9, v7 9.0 / 9.1.
- Wall-clock kernel times put v5 (G=2-4) and v7 at about 2x faster than v1, **but this is not supported by the fixed-clock profile**: under Nsight Compute (SM clock 585 MHz) v1 needs 10.09 M cycles, v5<2> 10.22 M and v7 10.77 M, with 573 M / 597 M / 648 M warp instructions. Same cycles means the same work rate, so the wall-clock gap is most likely the GPU clock not having ramped when v1 (always timed first) ran. Even in the profiled runs v5 G=2 varied between 12.0 and 14.9 ms against 8.35 ms earlier. Do not claim either speedup until re-measured with the burn-in added to the driver (`--no-burnin` restores the old behaviour).
- What does hold in both views: v2 (32 lanes per edge) is slower than v1 (wall clock 19-35 ms vs 13-18 ms, and 21.5 M vs 10.1 M cycles).
- v6 (thread per vertex) equals v1 on as-Skitter but is 6-8x slower on Wiki-Vote / email-Enron: only 8,298 / 36,692 threads cannot fill 40 SMs (each can hold 1,024 resident threads on Turing). Parallelism = n, not m.
- v4's hub-balancing hypothesis is not supported: max out-degree after orientation is 74 / 70 / 231, so there are no big hubs; v4 pays 10.3M tasks and 176-194 ms host task build (229-265 ms end-to-end).

**Replicated on a fresh Colab VM (session 2, same Tesla T4 model, built from `stage3` BEFORE the burn-in existed; logs `results/colab_run_2026-10-10_T4_session2_noburnin.log`, `results/ncu_inst_executed_session2_2026-10-10_T4.log`):** as-Skitter kernel-only v1 14.6, v2 29.2, v5<2> 8.0, v5<4> 8.1, v7 8.7 ms; ncu cycles and instruction counts identical to session 1 (v1 10.095 M cycles / 573,094,654 inst, v2 21.513 M / 1,498,761,114, v7 10.779 M / 647,517,249, v5<2> 10.218 M / 596,953,389). So the wall-clock gap is reproducible, not a one-off, and the burn-in explanation above is still an untested hypothesis. Direct test: profile v1, v5<2> and v7 with `ncu --clock-control none` (natural clock) and compare durations; and re-run main (with burn-in).

**Burn-in test (session 3, `main`, log `results/colab_run_session3_burnin_asSkitter_ncu_natural_clock.log`): the clock-ramp hypothesis is confirmed.** With the 1.5 s burn-in the as-Skitter kernel times (best of 10) were v1 8.96, v5<2> 8.16, v5<4> 8.12, v7 9.16, v2 18.96, v3 23.08, v4 20.87, v6 16.35 ms; v1 fell from 14.6 ms to 8.4-9.4 ms across three runs. So v1, v5 and v7 are about equal, and the earlier "v5/v7 are 2x faster" came from v1 running at a lower clock. Natural-clock Nsight Compute (`--clock-control none`): v7 12.95 M cycles at 1.59 GHz (8.13 ms), v5<2> 11.71 M cycles at 1.59 GHz (7.35 ms); the v1 profile caught the first (burn-in) launch at 1.02 GHz (11.93 M cycles, 11.69 ms), so it is not a clean natural-clock number. Cycle counts agree with the fixed-clock profile (all about 10-13 M). Treat all rankings above this paragraph as superseded by it. The full burn-in run on all three graphs (pageable and pinned, `results/colab_run_session3_burnin_full_T4.log`) is now in results/results.csv and the README tables: pageable and pinned kernel times agree (e.g. Wiki-Vote v1 0.139 ms in both), which they did not before the burn-in. as-Skitter kernel-only: v1 9.08, v5<2> 8.35, v5<4> 8.24, v7 8.89, v2 19.3, v3 23.2, v4 20.8, v6 16.7 ms; best end-to-end v7 pinned 13.35 ms. CPU times on that run: serial 1641 ms, OMP-2 1036 ms (the email-Enron pinned-run CPU times were 47 / 41 ms vs about 33 / 20 ms in other runs, a noisy VM; the tables use the best of all runs).

### Stage 4 (session 4, `stage4` code, logs `results/colab_run_session4_stage4_full_T4.log`, `results/ncu_v1_v8_v9_cugraph_attempt_session4_T4.log`)
- v8 (warp per vertex, shared-memory hash set, 512 slots): as-Skitter 12.6 ms vs v1 8.9 ms; com-LiveJournal 61.3 vs 56.0 ms. Nsight Compute (fixed clock): 806,320,876 warp instructions (v1 573,094,654), 15.20 M cycles (v1 10.08 M), achieved occupancy 59.5 % (theoretical 100 %, 16.38 KB static shared memory per block), L1/TEX 50.2 %, DRAM 5.4 %. It executes more instructions and does not reach full occupancy; the idea that the 512-slot clear per vertex is the main cost is a hypothesis (average out-degree is about 6.5, so most of the table work is clearing). Untested follow-up: smaller table (HASH_SLOTS=64 or 128; the fallback keeps it correct).
- v9 (edges sorted by work): kernel 9.0-9.1 ms (= v1) but 205-211 ms of host sort, end-to-end 225 ms (pinned). Nsight Compute: 471,452,412 warp instructions (-18 % vs v1), 10.39 M cycles (+3 %), L1/TEX 88.3 %, DRAM 17.1 % (v1 10.0 %). Less divergence (fewer instructions per edge) but the same cycles, so the L1/TEX pipeline is what limits it; sorting makes neighbouring threads touch less related memory, which raises DRAM traffic.
- com-LiveJournal (34.7M edges, max out-degree 524, 177,820,130 triangles): serial CPU 10,743 ms, OMP-2 6,362 ms; kernel-only v1 55.98, v5<4> 49.15 (best), v5<8> 51.11, v7 57.33, v8 61.32, v9 61.09, v2 69.59, v3 73.26, v4 75.95, v6 115.48 ms; pinned copy 23.9 ms vs 61.6 ms pageable; best end-to-end v7 pinned 70.57 ms. Same ranking as the smaller graphs. v5<4> is about 10-12 % faster than v1 on both large graphs in all runs.
- cuGraph reference: not obtained. After `pip install cugraph-cu12 cudf-cu12 --extra-index-url=...` on the CUDA 13 Colab image the script reported `cugraph unavailable: libcudf.so: cannot open shared object file`; pip also printed dependency conflicts (dask-cuda / rapids-dask-dependency, two CuPy packages). Next try: no pip install, use whatever RAPIDS the Colab image preinstalls.
- Overall: at these sizes every variant is limited by the L1/TEX pipeline; the profile-guided kernels (v8, v9) reduced either instructions or divergence but did not shorten the L1-bound critical path. Only v5 (G=4) is a repeatable, modest (about 10 %) improvement over the simple v1.

### Transfers
On as-Skitter the pageable H2D copy of v1's data (95.5 MB) took 21.3 ms, longer than the v1 kernel; with page-locked memory the same copy took 7.8 ms (derived: about 4.5 GB/s vs 12.3 GB/s). Best end-to-end in the pre-burn-in run is v7 pinned at 13.28 ms (its kernel time is the one in question above, so this will be re-measured) (kernel 9.09 + H2D 4.19) because it uploads only `rowptr` and `col`. For this problem size the copy is as important as the kernel. Pinning staging cost (memcpy into pinned memory) is not timed.

### Nsight Compute, as-Skitter, one launch each (basic set)
| metric | v1 tc_edge_per_thread | v2 tc_warp_per_edge |
|---|---|---|
| Duration under ncu | 17.25 ms | 36.77 ms |
| Elapsed cycles | 10.09 M | 21.51 M |
| DRAM throughput | 9.96 % | 5.21 % |
| L2 throughput | 13.41 % | 4.12 % |
| L1/TEX throughput | 89.79 % | 58.55 % |
| L1/TEX hit / L2 hit | 83.05 % / 90.15 % | 83.27 % / 87.13 % |
| Avg active threads per warp | 11.07 | 19.92 |
| Achieved occupancy | 77.8 % | 80.1 % |
| Warp cycles per issued instruction | 17.52 | 14.55 |

What this supports: neither kernel is DRAM-bandwidth-bound (DRAM 5-10 %, caches hit 83-90 %); v1 is limited by the L1/TEX pipeline (about 90 %) and loses two thirds of its lanes to divergence (11 of 32 active), yet v2 with better lane use (20 of 32) needs 2.1x the cycles. Both are stalled mostly on L1TEX scoreboard dependencies (46.6 % / 47.2 % of issue-to-issue cycles). Measured follow-up: v2 executes 1,498,761,114 warp instructions vs 573,094,654 for v1 (2.6x) for 2.1x the cycles, so v2's slowdown is extra instructions (binary-search steps, warp reduction, per-edge bookkeeping on short lists), not stalls. v7 adds 13 % instructions over v1 (647.5 M) for its binary search on `rowptr`; v5<2> adds 4 % (597.0 M). Raw values: results/ncu_summary_2026-10-10_T4.txt. Caveats: ncu pins the SM clock to 585 MHz ("SM Frequency" above), so absolute durations are inflated against the 13-17 ms measured in normal runs; compare ratios only. One launch each, no repeats.

### Effective-bandwidth estimate
`scripts/bandwidth_estimate.py` divides the minimum bytes (CSR arrays) by kernel time: 1-5 % of the T4's 320 GB/s peak. This is only a lower bound: lists are re-read by many edges, so real traffic is higher, and ncu shows DRAM throughput of 5-10 %. It should not be read as "wasting 95 % of bandwidth".

### Caveats
- CPU baseline: straightforward merge counter on the 2 cores Colab provides (cpu_omp2). GPU-vs-CPU speedups are against a modest baseline. CPU and GPU times both exclude file loading and orientation.
- Run-to-run variation on the shared T4 is real: v2 on as-Skitter ranged 19.1-27.9 ms across runs, and Wiki-Vote v1 was 0.26 ms in the pageable run and 0.13 ms in the pinned run. Treat sub-millisecond rows and differences under about 20 % as noise.
- Kernel time is best-of-5 warm (one launch under ncu).

### Next
- Full benchmark on all graphs (pageable + pinned) with the burn-in; the pre-burn-in rows are archived in results/results_pre_burnin.csv and results.csv is empty until the new run.
- A larger graph (e.g. com-LiveJournal) to see whether the ranking and the transfer share change.
- Optional: k-core peeling on the GPU for the triangle-density densest-subgraph problem.
