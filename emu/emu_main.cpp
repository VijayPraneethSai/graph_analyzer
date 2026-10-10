// emu_main.cpp - runs kernels.cuh on the CPU through cuda_shim.h and compares every kernel with the
// serial CPU count (itself checked against brute force by scripts/gen_test_graph.py).
// usage: emu_main graph.txt [expected_count]
// Uses deliberately tiny grids so grid-stride loops iterate several times, and tiny SM_CAP/TASK_CHUNK
// (set with -D at compile time) so the shared-memory fallback and multi-chunk tasks are exercised.
#include "cuda_shim.h"
#include "../src/graph.h"
#include "../src/kernels.cuh"

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    Graph g;
    if (!load_graph(argv[1], g)) return 1;
    uint64_t expected = cpu_triangles_serial(g);
    if (argc > 2 && (uint64_t)strtoull(argv[2], nullptr, 10) != expected) {
        printf("serial CPU count %llu != brute force %s\n", (unsigned long long)expected, argv[2]);
        return 1;
    }
    std::vector<uint32_t> te, ts;
    build_tasks(g, TASK_CHUNK, te, ts);
    uint32_t nt = (uint32_t)te.size();
    std::vector<uint32_t> perm;
    build_edge_order(g, perm);
    {   // perm must be a permutation of 0..m-1 (checked here so a bad sort cannot hide behind the count)
        std::vector<char> seen(g.m, 0);
        bool perm_ok = perm.size() == g.m;
        for (size_t i = 0; perm_ok && i < perm.size(); i++) { perm_ok = perm[i] < g.m && !seen[perm[i]]; if (perm_ok) seen[perm[i]] = 1; }
        if (!perm_ok) { printf("build_edge_order: NOT a permutation\n"); return 1; }
    }
    printf("HASH_SLOTS=%d (v8 fallback taken for out-degree > %d; this graph max out-degree=%u)\n", HASH_SLOTS,
           HASH_SLOTS / 2, g.max_out_degree);
    bool ok = true;
    auto check = [&](const char* name, unsigned long long got) {
        bool good = got == expected;
        printf("  %-22s %llu %s\n", name, got, good ? "ok" : "MISMATCH");
        ok &= good;
    };
    unsigned long long tot;
    const uint32_t *rp = g.rowptr.data(), *cl = g.col.data(), *sr = g.src.data();
    for (unsigned block : {32u, 64u, 96u}) {
        for (unsigned grid : {1u, 3u, 7u}) {
            printf("grid=%u block=%u  n=%u m=%u tasks=%u expected=%llu\n", grid, block, g.n, g.m, nt,
                   (unsigned long long)expected);
            tot = 0; emu_launch(grid, block, [&] { tc_edge_per_thread(rp, cl, sr, g.m, &tot); }); check("v1_edge_per_thread", tot);
            tot = 0; emu_launch(grid, block, [&] { tc_warp_per_edge(rp, cl, sr, g.m, &tot); });  check("v2_warp_per_edge", tot);
            tot = 0; emu_launch(grid, block, [&] { tc_block_smem(rp, cl, g.n, &tot); });         check("v3_block_smem", tot);
            tot = 0; emu_launch(grid, block, [&] { tc_task_warp(rp, cl, sr, te.data(), ts.data(), nt, &tot); }); check("v4_task_warp", tot);
            tot = 0; emu_launch(grid, block, [&] { tc_vertex_per_thread(rp, cl, g.n, &tot); });  check("v6_vertex_per_thread", tot);
            tot = 0; emu_launch(grid, block, [&] { tc_edge_find_src(rp, cl, g.n, g.m, &tot); }); check("v7_edge_find_src", tot);
            tot = 0; emu_launch(grid, block, [&] { tc_warp_vertex_hash(rp, cl, g.n, &tot); });  check("v8_warp_vertex_hash", tot);
            tot = 0; emu_launch(grid, block, [&] { tc_edge_sorted(rp, cl, sr, perm.data(), g.m, &tot); }); check("v9_edge_sorted", tot);
            tot = 0; emu_launch(grid, block, [&] { tc_group_per_edge<2>(rp, cl, sr, g.m, &tot); });  check("v5_g2", tot);
            tot = 0; emu_launch(grid, block, [&] { tc_group_per_edge<4>(rp, cl, sr, g.m, &tot); });  check("v5_g4", tot);
            tot = 0; emu_launch(grid, block, [&] { tc_group_per_edge<8>(rp, cl, sr, g.m, &tot); });  check("v5_g8", tot);
            tot = 0; emu_launch(grid, block, [&] { tc_group_per_edge<16>(rp, cl, sr, g.m, &tot); }); check("v5_g16", tot);
            tot = 0; emu_launch(grid, block, [&] { tc_group_per_edge<32>(rp, cl, sr, g.m, &tot); }); check("v5_g32", tot);
        }
    }
    printf(ok ? "ALL OK\n" : "FAILED\n");
    return ok ? 0 : 1;
}
