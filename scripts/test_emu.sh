#!/usr/bin/env bash
# CPU-only logic tests: runs every CUDA kernel through the thread-level emulator (emu/) on small
# graphs and compares against a brute-force triangle count. No GPU needed. Run before every Colab run.
set -e
cd "$(dirname "$0")/.."
mkdir -p data bin
g++ -O2 -std=c++20 -pthread -DSM_CAP=8 -DTASK_CHUNK=4 -DHASH_SLOTS=16 emu/emu_main.cpp -o bin/emu_small   # exercises smem fallback, multi-chunk tasks, v8 hash collisions + global-memory fallback (out-degree > 8)
g++ -O2 -std=c++20 -pthread                           emu/emu_main.cpp -o bin/emu_default
fail=0
run() {  # name generator args... out
  out="${@: -1}"; b=$(python3 scripts/$1.py "${@:2}")
  for bin in emu_small emu_default; do
    if timeout 300 ./bin/$bin "$out" "$b" | tail -1 | grep -q "ALL OK"; then echo "PASS $bin $out (brute force=$b)"; else echo "FAIL $bin $out"; fail=1; fi
  done
}
run gen_test_graph 300 0.05 1 data/t_er300.txt
run gen_test_graph 800 0.02 7 data/t_er800.txt
run gen_powerlaw 400 4 3 data/t_pl400.txt
run gen_powerlaw 900 6 5 data/t_pl900.txt

# Stage 5: densest-subgraph pipeline. Kernels tc_per_vertex_alive + mark_peel run through the emulator and are
# compared with a brute-force per-vertex count on random alive masks; the full peeling loop (real batch_peel_loop)
# must equal cpu_batch_peel exactly and be explained by independent oracles; --tiny checks the 3(1+eps)
# approximation guarantee against a brute-force search over all subsets (n <= 16).
# -fopenmp so that the OpenMP CPU variants are exercised too.
g++ -O2 -std=c++20 -pthread -fopenmp emu/emu_densest.cpp -o bin/emu_densest
for f in data/t_er300.txt data/t_er800.txt data/t_pl400.txt data/t_pl900.txt; do
  if timeout 600 ./bin/emu_densest "$f" | tail -1 | grep -q "ALL OK"; then echo "PASS emu_densest $f"; else echo "FAIL emu_densest $f"; fail=1; fi
done
if timeout 600 ./bin/emu_densest --tiny | tail -1 | grep -q "ALL OK"; then echo "PASS emu_densest --tiny (brute-force densest subgraph)"; else echo "FAIL emu_densest --tiny"; fail=1; fi
# Stage 6: paper algorithms (src/dsg.h) + the level-synchronous core-decomposition kernels (core_min_alive_tri,
# core_peel_level) run through the emulator inside the real host loop core_decomp_loop. --tiny compares Alg 3 core numbers,
# Exact (Alg 1), CoreExact (Alg 4), Greedy and Greedy++ with brute force over all vertex subsets on ~480 small graphs;
# file mode cross-checks larger graphs (Exact == CoreExact == pruned pipeline, emulated GPU cores == CPU cores).
g++ -O2 -std=c++20 -pthread -fopenmp emu/emu_dsg.cpp -o bin/emu_dsg
for f in data/t_er300.txt data/t_er800.txt data/t_pl400.txt data/t_pl900.txt; do
  if timeout 600 ./bin/emu_dsg "$f" | tail -1 | grep -q "ALL OK"; then echo "PASS emu_dsg $f"; else echo "FAIL emu_dsg $f"; fail=1; fi
done
if timeout 900 ./bin/emu_dsg --tiny | tail -1 | grep -q "ALL OK"; then echo "PASS emu_dsg --tiny (brute-force core numbers, Exact, CoreExact, Greedy++)"; else echo "FAIL emu_dsg --tiny"; fail=1; fi
if timeout 900 ./bin/emu_dsg --mid | tail -1 | grep -q "ALL OK"; then echo "PASS emu_dsg --mid (Exact as oracle for CoreExact: multi-component search and network shrinking)"; else echo "FAIL emu_dsg --mid"; fail=1; fi
if timeout 900 ./bin/emu_dsg --hub | tail -1 | grep -q "ALL OK"; then echo "PASS emu_dsg --hub (incremental core decomposition on hubs with > 32 neighbours)"; else echo "FAIL emu_dsg --hub"; fail=1; fi
[ $fail -eq 0 ] && echo "ALL EMULATOR TESTS PASSED" || { echo "SOME TESTS FAILED"; exit 1; }
