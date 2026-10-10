// cpu_tc.cpp - CPU baselines (single-thread and OpenMP).
// usage: cpu_tc <edge_list.txt> [repeats]
// prints one CSV line per implementation: impl,graph,n,m,triangles,time_ms (best of repeats)
#include "graph.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s graph.txt [repeats]\n", argv[0]); return 2; }
    int repeats = argc > 2 ? atoi(argv[2]) : 3;
    Graph g;
    double t0 = now_ms();
    if (!load_graph(argv[1], g)) return 1;
    fprintf(stderr, "loaded n=%u m=%u max_out_deg=%u in %.1f ms\n", g.n, g.m, g.max_out_degree, now_ms() - t0);

    uint64_t tri_s = 0, tri_p = 0;
    double best_s = 1e30, best_p = 1e30;
    for (int r = 0; r < repeats; r++) {
        double a = now_ms(); tri_s = cpu_triangles_serial(g); best_s = std::min(best_s, now_ms() - a);
        a = now_ms();        tri_p = cpu_triangles_omp(g);    best_p = std::min(best_p, now_ms() - a);
    }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("cpu_serial,%s,%u,%u,%llu,%.3f\n", argv[1], g.n, g.m, (unsigned long long)tri_s, best_s);
    printf("cpu_omp%d,%s,%u,%u,%llu,%.3f\n", threads, argv[1], g.n, g.m, (unsigned long long)tri_p, best_p);
    if (tri_s != tri_p) { fprintf(stderr, "MISMATCH serial vs omp\n"); return 1; }
    return 0;
}
