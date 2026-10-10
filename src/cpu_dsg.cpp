// cpu_dsg.cpp - CPU-only runner for the paper algorithms in dsg.h (triangle-density densest subgraph).
// usage: cpu_dsg graph.txt [--T 10] [--no-exact] [--no-full-exact] [--no-greedypp]
//   --T N              Greedy++ passes (default 10)
//   --no-exact         skip both flow-based algorithms (Exact = DSD Alg 1 and CoreExact = Alg 4)
//   --no-full-exact    skip only Exact (Alg 1, the slow one on big graphs); CoreExact still runs
//   --no-greedypp      skip Greedy++
// Output: one CSV row per algorithm, columns documented at dsg_row() in dsg.h:
//   impl,graph,n,m,triangles,vertices,density,time_ms,h2d_ms,total_ms,rounds,kmax
//   cpu_core_decomp (row = the kmax-core), cpu_greedy, cpu_greedy_pp, cpu_exact, cpu_core_exact
// Consistency checks (any failure => message on stderr, exit 1): Exact and CoreExact must give the same density as an exact
// rational; Greedy / Greedy++ must not exceed it; Greedy must reach at least 1/3 of it. Timings are for the machine this runs on.
#include "dsg.h"

int main(int argc, char** argv) {
    std::vector<const char*> pos;
    uint32_t T = 10;
    bool exact = true, full_exact = true, gpp = true;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--T") == 0 && i + 1 < argc) T = (uint32_t)std::max(1, atoi(argv[++i]));
        else if (strcmp(argv[i], "--no-exact") == 0) exact = false;
        else if (strcmp(argv[i], "--no-full-exact") == 0) full_exact = false;
        else if (strcmp(argv[i], "--no-greedypp") == 0) gpp = false;
        else pos.push_back(argv[i]);
    }
    if (pos.size() != 1) {
        fprintf(stderr, "usage: %s graph.txt [--T 10] [--no-exact] [--no-full-exact] [--no-greedypp]\n", argv[0]);
        return 2;
    }
    const char* path = pos[0];
    Graph g;
    if (!load_graph(path, g)) return 1;
    fprintf(stderr, "loaded n=%u m=%u max_out_deg=%u\n", g.n, g.m, g.max_out_degree);
    std::vector<std::string> rows;

    double t0 = now_ms();
    std::vector<uint32_t> core;
    uint32_t kmax = 0;
    if (!core_decomposition(g, core, &kmax)) { fprintf(stderr, "core number does not fit uint32\n"); return 1; }
    const double t_core = now_ms() - t0;
    const Dens kd = core_level_density(g, core, kmax);
    rows.push_back(dsg_row("cpu_core_decomp", path, g.n, g.m, kd.num, kd.den, t_core, 0, 0, kmax));

    t0 = now_ms();
    DsgResult gr = greedy(g);
    const double t_gr = now_ms() - t0;
    if (!gr.ok) { fprintf(stderr, "greedy failed: %s\n", gr.error.c_str()); return 1; }
    rows.push_back(dsg_row("cpu_greedy", path, g.n, g.m, gr.tri, gr.verts, t_gr, 0, g.n, 0));

    DsgResult gp;
    if (gpp) {
        t0 = now_ms();
        gp = greedy_pp(g, T);
        const double t_gp = now_ms() - t0;
        if (!gp.ok) { fprintf(stderr, "greedy_pp failed: %s\n", gp.error.c_str()); return 1; }
        if (dens_cmp(gp.dens(), gr.dens()) < 0) { fprintf(stderr, "MISMATCH: greedy_pp worse than greedy\n"); return 1; }
        rows.push_back(dsg_row("cpu_greedy_pp", path, g.n, g.m, gp.tri, gp.verts, t_gp, 0, T, 0));
    }

    if (exact) {
        DsgResult ce, ex;
        t0 = now_ms();
        ce = core_exact(g, core);
        const double t_ce = now_ms() - t0;
        if (!ce.ok) { fprintf(stderr, "core_exact failed: %s\n", ce.error.c_str()); return 1; }
        rows.push_back(dsg_row("cpu_core_exact", path, g.n, g.m, ce.tri, ce.verts, t_ce, 0, ce.flow_calls, 0));
        if (full_exact) {
            t0 = now_ms();
            ex = exact_densest(g);
            const double t_ex = now_ms() - t0;
            if (!ex.ok) { fprintf(stderr, "exact_densest failed: %s\n", ex.error.c_str()); return 1; }
            rows.push_back(dsg_row("cpu_exact", path, g.n, g.m, ex.tri, ex.verts, t_ex, 0, ex.flow_calls, 0));
            if (dens_cmp(ex.dens(), ce.dens()) != 0) {
                fprintf(stderr, "MISMATCH: exact %llu/%llu vs core_exact %llu/%llu\n", (unsigned long long)ex.tri, (unsigned long long)ex.verts,
                        (unsigned long long)ce.tri, (unsigned long long)ce.verts);
                return 1;
            }
        }
        if (dens_gt(gr.dens(), ce.dens()) || (gpp && dens_gt(gp.dens(), ce.dens()))) { fprintf(stderr, "MISMATCH: a heuristic exceeds the exact optimum\n"); return 1; }
        if (ce.verts && (u128)ce.tri * gr.verts > (u128)3 * gr.tri * ce.verts) { fprintf(stderr, "MISMATCH: greedy below 1/3 of the optimum\n"); return 1; }
        fprintf(stderr, "optimum %llu/%llu = %.6f   greedy %.6f   greedy++(T=%u) %.6f\n", (unsigned long long)ce.tri, (unsigned long long)ce.verts,
                dens_value(ce.dens()), dens_value(gr.dens()), T, gpp ? dens_value(gp.dens()) : -1.0);
    }
    for (auto& r : rows) puts(r.c_str());
    return 0;
}
