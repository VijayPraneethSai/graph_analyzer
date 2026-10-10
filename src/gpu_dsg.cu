// gpu_dsg.cu - densest subgraph (triangle density), paper algorithms with a GPU (k,Psi)-core decomposition.
// usage: gpu_dsg graph.txt [--T 10] [--repeats N] [--pinned] [--no-burnin] [--no-exact] [--no-greedy] [--no-greedypp]
//                          [--verify-full] [--verbose]
//   --T N          Greedy++ passes on the pruned graph (default 10)
//   --repeats N    timed GPU core-decomposition runs, best of N reported (default 5; one untimed verified run precedes)
//   --pinned       page-locked staging buffers (untimed copy) for the uploads; GPU rows get the suffix "_pinned"
//   --no-burnin    skip the ~1.5 s clock-ramp burn-in (on by default, as in gpu_densest)
//   --no-exact     stop after the GPU core decomposition, greedy and pruning (no CoreExact, no verify-full)
//   --no-greedy    skip the CPU greedy on the full graph (use on graphs with huge hubs); lb is then the kmax-core density only
//   --no-greedypp  skip Greedy++ on the pruned graph
//   --verify-full  also run CoreExact on the FULL graph with the CPU core numbers and require exactly the same optimum
//   --verbose      per-round diagnostics of one extra untimed GPU run (alive, triangles, level k, removed)
//
// WHAT RUNS WHERE (nothing else is on the GPU):
//   GPU: the level-synchronous (k,Psi)-core decomposition (kernels tc_per_vertex_alive, core_min_alive_tri, core_peel_level).
//   CPU: everything else - greedy (lower bound), pruning, CoreExact (Dinic max-flow), Greedy++.
// PIPELINE
//   (1) GPU core numbers; they must equal the CPU Alg 3 core numbers element-wise, else "MISMATCH", exit 1, no rows.
//   (2) lb = max(density of the kmax-core, greedy density): both are real subgraphs, so lb <= rho_opt. The densest subgraph
//       lies in the ceil(rho_opt)-core (DSD paper Lemma 7), hence in the ceil(lb)-core (cores nest) -> keep only vertices with
//       core >= ceil(lb) (prune_to_core in dsg.h).
//   (3) On the pruned graph: CoreExact (Alg 4) with the GPU core numbers restricted to the kept vertices (they ARE the core
//       numbers of the pruned graph, proof at prune_to_core; also asserted here against a fresh CPU decomposition of the
//       pruned graph), and Greedy++.
//   (4) --verify-full: CoreExact on the full graph with the CPU core numbers; densities must be equal as exact fractions.
//
// Output (stdout, only if everything verified): CSV rows, columns as in dsg_row() (src/dsg.h):
//   impl,graph,n,m,triangles,vertices,density,time_ms,h2d_ms,total_ms,rounds,kmax
//   cpu_core_decomp           serial Alg 3 on the full graph (single run); row = kmax-core
//   gpu_core_decomp[_pinned]  INCREMENTAL frontier version (default, kernels core_find/update/retire_frontier). time_ms = best of N
//                             runs of the whole level loop: ONE full triangle count, then per round memsets, 4 kernels and the 24-byte
//                             device-to-host copy per round (= the host sync); EXCLUDES uploads (h2d_ms = the three timed
//                             uploads rowptr+col+src+urow+ucol), burn-in, the final core[] download and the verification. rounds = number
//                             of level-synchronous rounds. Compare with cpu_core_decomp for kernel-only vs serial CPU.
//   gpu_core_decomp_recount[_pinned]  (--with-recount only) the first design: full triangle recount every round (kept as the baseline)
//   cpu_greedy                full graph; rounds = n removals
//   prune_kcore               time = building the pruned graph; n,m = PRUNED size; triangles,vertices = pruned graph;
//                             rounds = ceil(lb), kmax = max core number
//   cpu_core_exact_pruned_seeded  Alg 4 on the pruned graph, seeded with the Greedy++ result as lower bound (time excludes the
//                             decomposition and the Greedy++ run); rounds = max-flow calls
//   cpu_greedy_pp_pruned      Greedy++ on the pruned graph; rounds = T passes (run BEFORE CoreExact, its time IS in the pipeline totals)
//   pipeline_gpu_core[_pinned] h2d_ms = uploads; time_ms = gpu core (best) + greedy + prune + Greedy++(pruned) + seeded CoreExact(pruned)
//   pipeline_cpu_core         time_ms = cpu core + the same greedy + prune + CoreExact(pruned) (the identical CPU remainder, incl. Greedy++)
//   cpu_core_exact_full       (--verify-full only) Alg 4 on the full graph with CPU core numbers
// Graph loading, verification, burn-in and the final downloads are in none of the times. There is no CPU-OpenMP core
// decomposition here: Alg 3 is a sequential min-first peeling (see README).
#include <cuda_runtime.h>
#include <string>
#include "dsg.h"
#include "kernels.cuh"

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

static bool g_pinned = false;

// Allocates device memory, copies v to it and adds the (event-timed) copy time to *h2d_ms (same helper as gpu_densest.cu).
template <class T>
static T* upload(const std::vector<T>& v, double* h2d_ms, cudaEvent_t e0, cudaEvent_t e1) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(T)));
    const T* host = v.data();
    T* pin = nullptr;
    if (g_pinned && !v.empty()) {                      // staging copy is outside the timed region
        CUDA_CHECK(cudaMallocHost((void**)&pin, v.size() * sizeof(T)));
        memcpy(pin, v.data(), v.size() * sizeof(T));
        host = pin;
    }
    CUDA_CHECK(cudaEventRecord(e0));
    if (!v.empty()) CUDA_CHECK(cudaMemcpy(d, host, v.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    *h2d_ms += ms;
    if (pin) CUDA_CHECK(cudaFreeHost(pin));
    return d;
}

// Backend for core_decomp_loop (dsg.h).
struct GpuCoreBackend {
    uint32_t n, m;
    const uint32_t *rowptr, *col, *src;
    uint8_t* alive = nullptr;
    uint32_t* core = nullptr;
    uint32_t* tri = nullptr;
    unsigned* min_tri = nullptr;
    unsigned long long* stats = nullptr;       // stats[0] = T, stats[1] = removed this round, stats[2] = level k
    int block = 256, grid_edges = 1, grid_vertices = 1;

    void init(uint32_t n_, uint32_t m_, const uint32_t* rp, const uint32_t* cl, const uint32_t* sr) {
        n = n_; m = m_; rowptr = rp; col = cl; src = sr;
        CUDA_CHECK(cudaMalloc(&alive, std::max<size_t>(n, 1)));
        CUDA_CHECK(cudaMalloc(&core, std::max<size_t>(n, 1) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&tri, std::max<size_t>(n, 1) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&min_tri, sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&stats, 3 * sizeof(unsigned long long)));
        auto blocks_for = [&](uint64_t threads) {
            uint64_t b = (threads + block - 1) / block;
            return (int)std::max<uint64_t>(1, std::min<uint64_t>(b, 1u << 20));
        };
        grid_edges = blocks_for(m);
        grid_vertices = blocks_for(n);
    }
    void reset_state() {
        CUDA_CHECK(cudaMemset(alive, 1, n));
        CUDA_CHECK(cudaMemset(core, 0xFF, (size_t)n * sizeof(uint32_t)));          // UINT32_MAX = not assigned yet
    }
    void begin_round() {
        CUDA_CHECK(cudaMemset(tri, 0, (size_t)n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(stats, 0, 3 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(min_tri, 0xFF, sizeof(unsigned)));                   // 0xFFFFFFFF = "no alive vertex seen"
    }
    void run_tc() { tc_per_vertex_alive<<<grid_edges, block>>>(rowptr, col, src, alive, m, tri, stats); }
    void run_min() { core_min_alive_tri<<<grid_vertices, block>>>(tri, alive, n, min_tri); }
    void run_peel(uint64_t k) { core_peel_level<<<grid_vertices, block>>>(tri, alive, core, n, k, min_tri, stats + 1, stats + 2); }
    void read_stats(uint64_t& T, uint64_t& removed, uint64_t& k) {                // blocking copy = the per-round host sync
        unsigned long long h[3];
        CUDA_CHECK(cudaMemcpy(h, stats, sizeof h, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());
        T = h[0]; removed = h[1]; k = h[2];
    }
    void download_core(std::vector<uint32_t>& out) {
        out.resize(n);
        if (n) CUDA_CHECK(cudaMemcpy(out.data(), core, (size_t)n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
};

// Backend for core_decomp_loop_inc (dsg.h): the incremental frontier kernels of kernels.cuh. Needs the oriented CSR (for the ONE full
// triangle count that initialises tri[]) and the undirected sorted CSR (for the per-frontier updates).
struct GpuCoreBackendInc {
    uint32_t n, m;
    const uint32_t *rowptr, *col, *src, *urow, *ucol;
    uint8_t* state = nullptr;                  // 0 removed, 1 alive, 2 frontier of the current round
    uint32_t* core = nullptr;
    uint32_t* tri = nullptr;
    uint32_t* frontier = nullptr;              // vertex ids of the current frontier (each vertex enters once per run: n entries are enough)
    unsigned* min_tri = nullptr;
    unsigned long long* stats = nullptr;       // stats[0] = T (init), stats[1] = frontier size = removed this round, stats[2] = level k
    int block = 256, grid_edges = 1, grid_vertices = 1, grid_update = 1, grid_retire = 1;

    void init(uint32_t n_, uint32_t m_, const uint32_t* rp, const uint32_t* cl, const uint32_t* sr, const uint32_t* ur, const uint32_t* uc) {
        n = n_; m = m_; rowptr = rp; col = cl; src = sr; urow = ur; ucol = uc;
        CUDA_CHECK(cudaMalloc(&state, std::max<size_t>(n, 1)));
        CUDA_CHECK(cudaMalloc(&core, std::max<size_t>(n, 1) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&tri, std::max<size_t>(n, 1) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&frontier, std::max<size_t>(n, 1) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&min_tri, sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&stats, 3 * sizeof(unsigned long long)));
        auto blocks_for = [&](uint64_t threads) {
            uint64_t b = (threads + block - 1) / block;
            return (int)std::max<uint64_t>(1, std::min<uint64_t>(b, 1u << 20));
        };
        grid_edges = blocks_for(m);
        grid_vertices = blocks_for(n);
        // one warp per frontier vertex = 8 warps per 256-thread block; the kernel is grid-stride, so a fixed cap is fine and a small
        // frontier simply leaves most warps idle (a few microseconds), without a device-to-host copy to learn the frontier size first
        grid_update = (int)std::max<uint64_t>(1, std::min<uint64_t>(((uint64_t)n + 7) / 8, 4096));
        grid_retire = (int)std::max<uint64_t>(1, std::min<uint64_t>(((uint64_t)n + block - 1) / block, 1024));
    }
    void reset_state() {
        CUDA_CHECK(cudaMemset(state, 1, n));
        CUDA_CHECK(cudaMemset(core, 0xFF, (size_t)n * sizeof(uint32_t)));          // UINT32_MAX = not assigned yet
    }
    void init_tri(uint64_t& T) {               // the only full triangle count of a run; also the correctness check of the device count
        CUDA_CHECK(cudaMemset(tri, 0, (size_t)n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(stats, 0, 3 * sizeof(unsigned long long)));
        tc_per_vertex_alive<<<grid_edges, block>>>(rowptr, col, src, state, m, tri, stats);   // state==1 everywhere = "alive"
        unsigned long long h[3];
        CUDA_CHECK(cudaMemcpy(h, stats, sizeof h, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());
        T = h[0];
    }
    void begin_round() {
        CUDA_CHECK(cudaMemset(stats + 1, 0, 2 * sizeof(unsigned long long)));       // tri[] is NOT reset: it is kept up to date
        CUDA_CHECK(cudaMemset(min_tri, 0xFF, sizeof(unsigned)));
    }
    void run_min() { core_min_alive_tri<<<grid_vertices, block>>>(tri, state, n, min_tri); }
    void run_find(uint64_t k) { core_find_frontier<<<grid_vertices, block>>>(tri, state, core, n, k, min_tri, frontier, stats + 1, stats + 2); }
    void run_update() { core_update_frontier<<<grid_update, block>>>(urow, ucol, state, frontier, stats + 1, tri); }
    void run_retire() { core_retire_frontier<<<grid_retire, block>>>(state, frontier, stats + 1); }
    void read_stats(uint64_t& removed, uint64_t& k) {                               // blocking copy = the per-round host sync
        unsigned long long h[3];
        CUDA_CHECK(cudaMemcpy(h, stats, sizeof h, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());
        removed = h[1]; k = h[2];
    }
    void download_core(std::vector<uint32_t>& out) {
        out.resize(n);
        if (n) CUDA_CHECK(cudaMemcpy(out.data(), core, (size_t)n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
};

int main(int argc, char** argv) {
    std::vector<const char*> pos;
    bool recount = false, burnin = true, exact = true, greedy_on = true, gpp = true, verify_full = false, verbose = false;
    int repeats = 5;
    uint32_t T_pass = 10;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--pinned") == 0) g_pinned = true;
        else if (strcmp(argv[i], "--with-recount") == 0) recount = true;
        else if (strcmp(argv[i], "--no-burnin") == 0) burnin = false;
        else if (strcmp(argv[i], "--no-exact") == 0) exact = false;
        else if (strcmp(argv[i], "--no-greedy") == 0) greedy_on = false;
        else if (strcmp(argv[i], "--no-greedypp") == 0) gpp = false;
        else if (strcmp(argv[i], "--verify-full") == 0) verify_full = true;
        else if (strcmp(argv[i], "--verbose") == 0) verbose = true;
        else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) repeats = std::max(1, atoi(argv[++i]));
        else if (strcmp(argv[i], "--T") == 0 && i + 1 < argc) T_pass = (uint32_t)std::max(1, atoi(argv[++i]));
        else pos.push_back(argv[i]);
    }
    if (pos.size() != 1) {
        fprintf(stderr, "usage: %s graph.txt [--T 10] [--repeats N] [--pinned] [--no-burnin] [--no-exact] [--no-greedy] [--no-greedypp] [--verify-full] [--verbose]\n", argv[0]);
        return 2;
    }
    const char* path = pos[0];

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "GPU: %s, SMs=%d, cc=%d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    Graph g;
    if (!load_graph(path, g)) return 1;
    fprintf(stderr, "loaded n=%u m=%u max_out_deg=%u\n", g.n, g.m, g.max_out_degree);
    const uint64_t T_full = cpu_triangles_serial(g);

    std::vector<std::string> rows;
    // ---- CPU reference core decomposition (Alg 3): decides whether the GPU result is accepted ----
    std::vector<uint32_t> core_cpu;
    uint32_t kmax = 0;
    double t0 = now_ms();
    if (!core_decomposition(g, core_cpu, &kmax)) { fprintf(stderr, "core number does not fit uint32\n"); return 1; }
    const double t_cpu_core = now_ms() - t0;
    const Dens kcore = core_level_density(g, core_cpu, kmax);
    rows.push_back(dsg_row("cpu_core_decomp", path, g.n, g.m, kcore.num, kcore.den, t_cpu_core, 0, 0, kmax));

    // ---- GPU: upload once ----
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    double h2d_ms = 0;
    uint32_t* d_rowptr = upload(g.rowptr, &h2d_ms, e0, e1);
    uint32_t* d_col = upload(g.col, &h2d_ms, e0, e1);
    uint32_t* d_src = upload(g.src, &h2d_ms, e0, e1);
    // undirected sorted CSR for the incremental update kernel (built on the host, uploaded once; its copy time is part of h2d_ms)
    std::vector<uint32_t> urow_h, ucol_h;
    build_undirected_csr(g, urow_h, ucol_h);
    uint32_t* d_urow = upload(urow_h, &h2d_ms, e0, e1);
    uint32_t* d_ucol = upload(ucol_h, &h2d_ms, e0, e1);
    GpuCoreBackendInc bi;
    bi.init(g.n, g.m, d_rowptr, d_col, d_src, d_urow, d_ucol);
    GpuCoreBackend be;                          // recount baseline (--with-recount only; measured on the first T4 run, see README)
    if (recount) be.init(g.n, g.m, d_rowptr, d_col, d_src);

    // Burn-in (untimed): ~1.5 s of GPU work so the SM clock is ramped before any timed run (see README, stage 3).
    if (burnin) {
        bi.reset_state();
        const double tb = now_ms();
        uint64_t Tb = 0;
        do { bi.init_tri(Tb); } while (now_ms() - tb < 1500.0);      // init_tri ends with a blocking copy, so each pass is synchronised
        CUDA_CHECK(cudaGetLastError());
    }

    bool ok = true;
    CoreLoopResult last;
    std::vector<uint32_t> core_gpu;
    // One full level loop; downloads core[] (untimed) and demands element-wise equality with the CPU Alg 3 result.
    auto run_once = [&](bool inc, bool diag, double* ms_out) {
        CUDA_CHECK(cudaDeviceSynchronize());
        const double ts = now_ms();
        CoreLoopResult r = inc ? core_decomp_loop_inc(g.n, T_full, bi, diag, stderr) : core_decomp_loop(g.n, T_full, be, diag, stderr);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (ms_out) *ms_out = now_ms() - ts;
        if (!r.ok) { fprintf(stderr, "GPU core decomposition error: %s\n", r.error.c_str()); ok = false; return r; }
        if (inc) bi.download_core(core_gpu); else be.download_core(core_gpu);
        if (core_gpu != core_cpu || r.kmax != kmax) {
            size_t bad = 0;
            while (bad < core_gpu.size() && core_gpu[bad] == core_cpu[bad]) bad++;
            fprintf(stderr, "MISMATCH gpu core numbers vs CPU Alg 3: kmax %u vs %u, first differing vertex %zu (gpu %u, cpu %u)\n", r.kmax, kmax,
                    bad, bad < core_gpu.size() ? core_gpu[bad] : 0u, bad < core_cpu.size() ? core_cpu[bad] : 0u);
            ok = false;
        }
        return r;
    };
    run_once(true, false, nullptr);                                  // untimed warm-up + verification (incremental, the default)
    if (verbose && ok) { fprintf(stderr, "--- per-round diagnostics (extra untimed run) ---\n"); run_once(true, true, nullptr); }
    double best_ms = 1e300;
    for (int i = 0; i < repeats && ok; i++) {
        double ms = 0;
        last = run_once(true, false, &ms);
        best_ms = std::min(best_ms, ms);
    }
    if (!ok) { fprintf(stderr, "VERIFICATION FAILED - no timing rows printed\n"); return 1; }
    const std::string gpu_name = std::string("gpu_core_decomp") + (g_pinned ? "_pinned" : "");
    rows.push_back(dsg_row(gpu_name.c_str(), path, g.n, g.m, kcore.num, kcore.den, best_ms, h2d_ms, last.rounds, kmax));
    fprintf(stderr, "GPU incremental core numbers verified element-wise against CPU Alg 3 (kmax %u, %u rounds)\n", kmax, last.rounds);
    if (recount) {                                                   // old design, for the comparison in the README (verified the same way)
        run_once(false, false, nullptr);
        double best_rc = 1e300;
        CoreLoopResult lrc;
        for (int i = 0; i < repeats && ok; i++) {
            double ms = 0;
            lrc = run_once(false, false, &ms);
            best_rc = std::min(best_rc, ms);
        }
        if (!ok) { fprintf(stderr, "VERIFICATION FAILED (recount) - no timing rows printed\n"); return 1; }
        const std::string rn = std::string("gpu_core_decomp_recount") + (g_pinned ? "_pinned" : "");
        rows.push_back(dsg_row(rn.c_str(), path, g.n, g.m, kcore.num, kcore.den, best_rc, h2d_ms, lrc.rounds, kmax));
    }

    // ---- (2) lower bound and pruning ----
    double t_greedy = 0;
    Dens lb = kcore;
    if (greedy_on) {
        t0 = now_ms();
        DsgResult gr = greedy(g);
        t_greedy = now_ms() - t0;
        if (!gr.ok) { fprintf(stderr, "greedy failed: %s\n", gr.error.c_str()); return 1; }
        rows.push_back(dsg_row("cpu_greedy", path, g.n, g.m, gr.tri, gr.verts, t_greedy, 0, g.n, 0));
        if (dens_gt(gr.dens(), lb)) lb = gr.dens();
    }
    const uint64_t kprune = dens_ceil(lb);
    Graph gp;
    std::vector<uint32_t> old_id, core_p;
    t0 = now_ms();
    prune_to_core(g, core_gpu, kprune, gp, old_id, core_p);          // GPU core numbers, restricted
    const double t_prune = now_ms() - t0;
    const uint64_t T_pruned = cpu_triangles_serial(gp);
    rows.push_back(dsg_row("prune_kcore", path, gp.n, gp.m, T_pruned, gp.n, t_prune, 0, kprune, kmax));
    fprintf(stderr, "lower bound %llu/%llu = %.6f -> keep core >= %llu: pruned graph n=%u m=%u (of n=%u m=%u)\n", (unsigned long long)lb.num,
            (unsigned long long)lb.den, dens_value(lb), (unsigned long long)kprune, gp.n, gp.m, g.n, g.m);
    {   // the restricted GPU core numbers must be the core numbers of the pruned graph (claim in prune_to_core), checked here
        std::vector<uint32_t> chk;
        if (!core_decomposition(gp, chk) || chk != core_p) { fprintf(stderr, "MISMATCH: restricted core numbers != core numbers of the pruned graph\n"); return 1; }
    }

    // ---- (3) CoreExact and Greedy++ on the pruned graph ----
    DsgResult ex_p, gpr;
    double t_exact = 0, t_gpp = 0;
    bool have_gpp = false;
    if (gpp || exact) {   // Greedy++ first: its result is the lower-bound seed for CoreExact (exact stays exact, see core_exact)
        t0 = now_ms();
        gpr = greedy_pp(gp, T_pass);
        t_gpp = now_ms() - t0;
        if (!gpr.ok) { fprintf(stderr, "greedy_pp failed: %s\n", gpr.error.c_str()); return 1; }
        have_gpp = true;
        if (gpp) rows.push_back(dsg_row("cpu_greedy_pp_pruned", path, gp.n, gp.m, gpr.tri, gpr.verts, t_gpp, 0, T_pass, 0));
    }
    if (exact) {
        t0 = now_ms();
        ex_p = core_exact(gp, core_p, true, have_gpp ? &gpr : nullptr);
        t_exact = now_ms() - t0;
        if (!ex_p.ok) { fprintf(stderr, "core_exact (pruned) failed: %s\n", ex_p.error.c_str()); return 1; }
        rows.push_back(dsg_row("cpu_core_exact_pruned_seeded", path, gp.n, gp.m, ex_p.tri, ex_p.verts, t_exact, 0, ex_p.flow_calls, 0));
        if (dens_cmp(ex_p.dens(), lb) < 0) { fprintf(stderr, "MISMATCH: exact optimum below the lower bound\n"); return 1; }
        if (have_gpp && dens_gt(gpr.dens(), ex_p.dens())) { fprintf(stderr, "MISMATCH: greedy++ exceeds the exact optimum\n"); return 1; }
    }
    if (exact) {
        const std::string pn = std::string("pipeline_gpu_core") + (g_pinned ? "_pinned" : "");
        rows.push_back(dsg_row(pn.c_str(), path, g.n, g.m, ex_p.tri, ex_p.verts, best_ms + t_greedy + t_prune + t_gpp + t_exact, h2d_ms, last.rounds, kmax));
        rows.push_back(dsg_row("pipeline_cpu_core", path, g.n, g.m, ex_p.tri, ex_p.verts, t_cpu_core + t_greedy + t_prune + t_gpp + t_exact, 0, 0, kmax));
    }

    // ---- (4) verification of the pruning against the unpruned algorithm ----
    if (exact && verify_full) {
        t0 = now_ms();
        DsgResult ex_f = core_exact(g, core_cpu);
        const double t_full = now_ms() - t0;
        if (!ex_f.ok) { fprintf(stderr, "core_exact (full) failed: %s\n", ex_f.error.c_str()); return 1; }
        if (dens_cmp(ex_f.dens(), ex_p.dens()) != 0) {
            fprintf(stderr, "MISMATCH: full-graph CoreExact %llu/%llu vs pruned pipeline %llu/%llu\n", (unsigned long long)ex_f.tri,
                    (unsigned long long)ex_f.verts, (unsigned long long)ex_p.tri, (unsigned long long)ex_p.verts);
            return 1;
        }
        rows.push_back(dsg_row("cpu_core_exact_full", path, g.n, g.m, ex_f.tri, ex_f.verts, t_full, 0, ex_f.flow_calls, 0));
        fprintf(stderr, "verify-full: optimum identical (%llu/%llu) with and without pruning\n", (unsigned long long)ex_f.tri, (unsigned long long)ex_f.verts);
    }

    for (auto& r : rows) puts(r.c_str());
    fflush(stdout);
    if (exact) fprintf(stderr, "optimum %llu/%llu = %.6f\n", (unsigned long long)ex_p.tri, (unsigned long long)ex_p.verts, dens_value(ex_p.dens()));
    return 0;
}
