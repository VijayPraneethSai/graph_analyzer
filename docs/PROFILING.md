# Profiling on Colab (and what to do if you cannot)

Rule: only claim a profiler result you actually obtained. Paste real output into `results/` and quote it; never fill in numbers from memory or expectation.

## 1. Check the environment
    nvidia-smi                 # GPU model, driver, memory; record the model with every result
    nvcc --version
    which ncu nsys             # are the tools even installed?
    ncu --version
    nsys --version

Colab runtimes vary. Try a tiny profile first:

    ncu --set basic ./bin/gpu_tc data/test800.txt 1

If you see `ERR_NVGPUCTRPERM` (or "The user does not have permission to access NVIDIA GPU Performance Counters"), Nsight Compute is blocked on that runtime: hardware counters are not available to you. Reconnecting to a different runtime or GPU type may or may not help; do not spend long on it. `nsys` (timeline only, no counters) often still works.

## 2. Commands
Per-kernel counters (one profile per variant, change `vN`):

    ncu --set full -o results/profile_vN ./bin/gpu_tc data/as-Skitter.txt 1

Profiling the newer kernels (v8 and v9) by kernel name, one launch each, instruction count included
(compare `smsp__inst_executed.sum` with v1/v2's profile; these have not been run yet):

    /usr/local/cuda/bin/ncu --set basic --metrics smsp__inst_executed.sum -k tc_warp_vertex_hash -c 1 ./bin/gpu_tc data/as-Skitter.txt 1
    /usr/local/cuda/bin/ncu --set basic --metrics smsp__inst_executed.sum -k tc_edge_sorted -c 1 ./bin/gpu_tc data/as-Skitter.txt 1

For v8 also look at shared-memory bank conflicts (Memory Workload section, "shared load/store bank conflicts") and
shared memory per block (occupancy limiter); for v9 compare "avg. active threads per warp" against v1's 11.1 of 32.

Timeline (kernel vs memcpy overlap, launch gaps, total GPU time):

    nsys profile -o results/timeline ./bin/gpu_tc data/as-Skitter.txt 1

Use repeat count 1 (last argument) so the profile is not huge. Open `.ncu-rep` / `.nsys-rep` files in the Nsight GUI on your laptop, or print a summary with `ncu --import results/profile_vN.ncu-rep --page details`.

## 3. Metrics to look at
| Metric (Nsight Compute section) | What it tells you |
|---|---|
| Achieved occupancy (Occupancy) | Active warps / max warps per SM. Low occupancy limits latency hiding; the cause may be registers, shared memory per block (v3), or too little work. High occupancy alone does not mean fast. |
| DRAM throughput % (Memory Workload / SOL) | How close you are to peak memory bandwidth. Triangle counting is usually memory-bound, so high DRAM % means you are near the limit of this access pattern. |
| L1/L2 hit rate | Whether neighbour-list reads are reused from cache. Binary searches revisit the same elements, so hits are expected; low hit rate on hub-heavy graphs points to scattered access. |
| Warp execution efficiency / branch divergence (Warp State, "avg. active threads per warp") | Fraction of the 32 lanes doing useful work. Low values mean divergence or short lists leaving lanes idle (v1 is expected to suffer; check rather than assume). |
| Global load efficiency, sectors per request (Memory Workload) | Coalescing. Ideal is few 32-byte sectors per warp request; many sectors per request means each lane touches a different cache line. |
| Atomics (atomic instruction count, memory throughput) | Contention on the single result counter. v2 does one atomic per warp, v3 one per block; compare against v1. |
| Stall reasons (Warp State) | Dominant reason warps wait (long scoreboard = waiting on global memory, etc.). Good for explaining the bottleneck. |

## 4. If counters are blocked
You can still make honest, smaller claims using only the program's own output:

1. **Kernel time vs host-to-device time.** `time_ms` (kernel) and `h2d_ms` are printed separately. Report both; if h2d is comparable to the kernel, say end-to-end speedup is limited by transfer.
2. **Estimated achieved bandwidth.** Compute bytes touched analytically (for example sum over oriented edges of the list elements read, times 4 bytes, plus rowptr/col traffic) divided by kernel time. Label it clearly as an analytical estimate, not a measurement, and note it ignores cache reuse. Compare to the GPU's published peak bandwidth only as context.
3. **Compare variants that differ only in access pattern** (e.g. v1 vs v2, or v3 with and without the shared-memory path via SM_CAP). Keep the graph, GPU and counts identical; the difference in time is evidence about the access pattern, but state it as a hypothesis, not proven cause.
4. **Hub-heavy vs uniform graphs.** Time the same kernel on as-Skitter (heavy-tailed degrees) and on a more uniform graph (e.g. a synthetic one from `scripts/gen_test_graph.py`, or Wiki-Vote). Normalise by edges or by estimated work. Differences hint at load imbalance; v4 exists to address that.
5. Run each configuration several times (the benchmark script already repeats) and report the best (min) as `make_table.py` does, noting that Colab is shared and noisy.

## 5. Resume honesty
Do not list Nsight Compute, Nsight Systems, or any other tool on the resume or in interviews unless you ran it successfully and can explain its output. If counters were blocked, say so and describe the measurements from section 4 instead.

## 6. Densest-subgraph pipeline (stage 5; commands only, NOT YET RUN)
Profile the two new kernels on as-Skitter (use `--no-greedy`, `--repeats 1`, `--no-burnin` so the profile is small; kernels are launched once per round, so select launches with `-c` / `--launch-skip`):

    # round 0 (whole graph alive, heaviest launch) of the triangle-participation kernel
    ncu --set full -k tc_per_vertex_alive -c 1 -o results/profile_tc_per_vertex_alive_r0 ./bin/gpu_densest data/as-Skitter.txt 1 10 --no-greedy --no-burnin --repeats 1
    # a late round (shrinking workload): skip the first launches. NOTE: the run does one untimed verified pass first, so
    # launches of the first pass come first; use --verbose to see how many rounds there are, then choose --launch-skip
    ncu --set full -k tc_per_vertex_alive --launch-skip 5 -c 1 -o results/profile_tc_per_vertex_alive_late ./bin/gpu_densest data/as-Skitter.txt 1 10 --no-greedy --no-burnin --repeats 1
    # the peel kernel (tiny, memory-bound over n elements)
    ncu --set full -k mark_peel -c 1 -o results/profile_mark_peel_r0 ./bin/gpu_densest data/as-Skitter.txt 1 10 --no-greedy --no-burnin --repeats 1
    # cheap counter-only checks (work if full-set counters are blocked)
    ncu --metrics smsp__inst_executed.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum -k tc_per_vertex_alive -c 1 ./bin/gpu_densest data/as-Skitter.txt 1 10 --no-greedy --no-burnin --repeats 1
    # timeline: kernel vs host-sync gaps between rounds
    nsys profile -o results/timeline_densest ./bin/gpu_densest data/as-Skitter.txt 1 10 --no-greedy --repeats 1

What to look at: for `tc_per_vertex_alive`, atomic throughput and L2 atomic stalls (hub contention on `tri[]`), "avg. active threads per warp" (compare with v1's 11.1 of 32 in the triangle stage), and how much of the launch is threads that exit after the two `alive[]` loads in late rounds (achieved occupancy vs. useful work). For `mark_peel`: DRAM throughput over `n` elements; it should be bandwidth-bound and tiny. In the nsys timeline, look at the idle gaps between rounds caused by the 16-byte device-to-host copy. Treat every statement as a hypothesis until the report is in hand.
