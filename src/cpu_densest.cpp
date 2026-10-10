// cpu_densest.cpp - CPU-only runner for the triangle-density densest-subgraph algorithms (no GPU needed).
// usage: cpu_densest graph.txt [eps_num eps_den] [--no-greedy]
// Runs cpu_batch_peel (serial, and OpenMP when built with -fopenmp), checks that they agree exactly, then
// cpu_greedy_peel. Prints the same CSV columns as gpu_densest (h2d_ms = 0), one run per row (not best-of-N).
// Useful for local sanity checks and as the "cpu-only" path of scripts/run_bench.sh --densest.
#include "densest.h"

int main(int argc, char** argv) {
    std::vector<const char*> pos;
    bool greedy = true;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-greedy") == 0) greedy = false;
        else pos.push_back(argv[i]);
    }
    if (pos.empty() || pos.size() == 2 || pos.size() > 3) {
        fprintf(stderr, "usage: %s graph.txt [eps_num eps_den] [--no-greedy]\n", argv[0]);
        return 2;
    }
    const char* path = pos[0];
    const uint64_t num = pos.size() == 3 ? strtoull(pos[1], nullptr, 10) : 1;
    const uint64_t den = pos.size() == 3 ? strtoull(pos[2], nullptr, 10) : 10;
    Graph g;
    if (!load_graph(path, g)) return 1;
    fprintf(stderr, "loaded n=%u m=%u max_out_deg=%u eps=%llu/%llu\n", g.n, g.m, g.max_out_degree,
            (unsigned long long)num, (unsigned long long)den);
    std::vector<std::string> rows;
    double t0 = now_ms();
    PeelResult ref = cpu_batch_peel(g, num, den, false);
    double t_serial = now_ms() - t0;
    if (!ref.ok) { fprintf(stderr, "cpu_batch_peel failed: %s\n", ref.error.c_str()); return 1; }
    rows.push_back(densest_row("cpu_batch_peel", path, g.n, g.m, ref.best_num, ref.best_den, ref.rounds, t_serial, 0));
#ifdef _OPENMP
    t0 = now_ms();
    PeelResult omp = cpu_batch_peel(g, num, den, true);
    double t_omp = now_ms() - t0;
    if (!omp.ok || !(omp == ref)) { fprintf(stderr, "MISMATCH: OpenMP batch peel differs from serial\n"); return 1; }
    rows.push_back(densest_row("cpu_batch_peel_omp", path, g.n, g.m, omp.best_num, omp.best_den, omp.rounds, t_omp, 0));
#endif
    if (greedy) {
        t0 = now_ms();
        GreedyResult gr = cpu_greedy_peel(g);
        double t_greedy = now_ms() - t0;
        if (!gr.ok) { fprintf(stderr, "cpu_greedy_peel failed: %s\n", gr.error.c_str()); return 1; }
        rows.push_back(densest_row("cpu_greedy_peel", path, g.n, g.m, gr.best_num, gr.best_den, g.n, t_greedy, 0));
        fprintf(stderr, "density ratio batch/greedy = %.6f\n",
                density_value(gr.best_num, gr.best_den) > 0
                    ? density_value(ref.best_num, ref.best_den) / density_value(gr.best_num, gr.best_den) : 0.0);
    }
    for (auto& r : rows) puts(r.c_str());
    return 0;
}
