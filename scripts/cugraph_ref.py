#!/usr/bin/env python3
"""cugraph_ref.py - reference GPU triangle count from NVIDIA cuGraph, to compare against our kernels.

usage: python3 scripts/cugraph_ref.py <edge_list.txt> [repeats]      (default repeats = 5)

Output (stdout, one line):  cugraph_triangle_count,<graph>,<n>,<m>,<triangles>,<time_ms>
  n = max vertex id + 1 (same convention as gpu_tc/cpu_tc), m = undirected edges after removing self loops
  and duplicates, time_ms = best of `repeats` timed calls after one untimed warm-up call, each call ended
  with a CUDA device synchronize.
  !! Load / graph-construction time is EXCLUDED (only cugraph.triangle_count is timed), so this is NOT
  directly comparable with our end-to-end numbers (which include host-to-device copies); compare it with
  our kernel-only time and say so. A note to that effect is also printed on stderr.

How the count is obtained (ASSUMPTION, verify against our CPU count before trusting the timing):
  cugraph.triangle_count(G) returns one row per vertex with the number of triangles that vertex belongs to.
  Every triangle has 3 vertices, so it is counted once at each of them: total triangles = sum(counts) / 3.
  The script asserts that the sum is divisible by 3 and exits 4 otherwise. The result must equal the
  `triangles` column printed by ./bin/cpu_tc and ./bin/gpu_tc; if it does not, the assumption or the graph
  construction is wrong - do not report the number.

Install hints for Colab (MAY NOT WORK: cuGraph wheels need a matching CUDA 12 / Python version and the
free T4 runtime image changes; check the pip output, nothing here is guaranteed):
    pip install cugraph-cu12 cudf-cu12 --extra-index-url=https://pypi.nvidia.com
If the import fails the script prints 'cugraph unavailable: <error>' to stderr and exits with code 3.
"""
import sys
import time


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = sys.argv[1]
    repeats = max(1, int(sys.argv[2])) if len(sys.argv) > 2 else 5

    try:
        import cudf
        import cugraph
    except Exception as e:  # ImportError, or CUDA/driver errors raised during import
        print(f"cugraph unavailable: {e}", file=sys.stderr)
        return 3

    def sync():
        try:
            import cupy
            cupy.cuda.Device().synchronize()
        except Exception:
            from numba import cuda
            cuda.synchronize()

    # --- load: SNAP format, whitespace separated "u v", lines starting with '#' are comments
    df = cudf.read_csv(path, sep=r"\s+", comment="#", header=None, names=["src", "dst"],
                       dtype=["int64", "int64"])
    df = df[df["src"] != df["dst"]]                                   # drop self loops
    lo = df[["src", "dst"]].min(axis=1)
    hi = df[["src", "dst"]].max(axis=1)
    df = cudf.DataFrame({"src": lo, "dst": hi}).drop_duplicates()     # one row per undirected edge
    n = int(max(df["src"].max(), df["dst"].max())) + 1
    m = len(df)

    G = cugraph.Graph(directed=False)       # undirected: cuGraph symmetrises the edge list internally
    G.from_cudf_edgelist(df, source="src", destination="dst", renumber=True)

    def run():
        res = cugraph.triangle_count(G)     # cudf.DataFrame: vertex, counts (triangles through that vertex)
        total = int(res["counts"].sum())
        sync()
        return total

    total3 = run()                           # untimed warm-up (also gives the count)
    best = float("inf")
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        t3 = run()                           # run() ends with a device sync
        best = min(best, (time.perf_counter() - t0) * 1000.0)
        total3 = t3
    if total3 % 3 != 0:
        print(f"ERROR: sum of per-vertex counts {total3} is not divisible by 3; the 'sum/3' assumption "
              f"does not hold for this cuGraph version", file=sys.stderr)
        return 4
    triangles = total3 // 3

    print("NOTE: load and graph-construction time is EXCLUDED from time_ms; only cugraph.triangle_count "
          "(+ per-vertex sum) is timed. Compare with kernel-only times, not end-to-end.", file=sys.stderr)
    print(f"cugraph_triangle_count,{path},{n},{m},{triangles},{best:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
