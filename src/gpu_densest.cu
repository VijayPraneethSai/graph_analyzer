// gpu_densest.cu - GPU batch peeling for the TRIANGLE-DENSITY DENSEST SUBGRAPH problem.
// usage: gpu_densest graph.txt [eps_num eps_den] [--pinned] [--no-burnin] [--repeats N] [--no-greedy] [--verbose]
//   eps = eps_num/eps_den (default 1/10): a vertex is peeled when its triangle participation is at most
//   (1+eps) * average. Algorithm, guarantee and arithmetic bounds: see src/densest.h and the README section.
//   --pinned     copy host arrays into page-locked memory (untimed) before the timed uploads; the GPU row gets
//                the suffix "_pinned" so pinned and pageable rows never mix.
//   --no-burnin  skip the ~1.5 s clock-ramp burn-in (the burn-in is on by default, see README: without it the
//                first timed run can execute at a lower SM clock).
//   --repeats N  number of timed GPU runs, best-of-N is reported (default 5; one untimed verified run precedes).
//   --no-greedy  skip cpu_greedy_peel. USE THIS on graphs with huge hubs / many triangles (e.g. as-Skitter,
//                com-LiveJournal): the sequential greedy re-intersects neighbour lists for every removal and can
//                take very long; no greedy row and no density ratio is printed then.
//   --verbose    per-round diagnostics on stderr (alive count, triangles T, removed, kernel ms), from one extra
//                untimed run that records CUDA events around the two kernels of each round.
//
// Output (stdout, only if EVERYTHING verified): CSV rows
//   impl,graph,n,m,best_triangles,best_vertices,density,rounds,time_ms,h2d_ms,total_ms
//   cpu_batch_peel, cpu_batch_peel_omp (only if built with OpenMP), cpu_greedy_peel, gpu_batch_peel[_pinned]
// time_ms of gpu_batch_peel is the best of N runs of exactly this: reset of alive/removed_round (2 cudaMemset) +
//   for each round { cudaMemset tri and counters, tc_per_vertex_alive, mark_peel, 16-byte device-to-host copy of
//   {T, removed} (= the host sync), host best-density bookkeeping }, measured with the host wall clock between two
//   cudaDeviceSynchronize calls. It EXCLUDES: graph loading, uploads (reported as h2d_ms), burn-in, the final
//   download of removed_round and the verification. The per-round host sync IS included, deliberately: it is part of
//   this design. h2d_ms = the three timed uploads rowptr + col + src. total_ms = time_ms + h2d_ms.
//   CPU rows are single runs (not best-of-N) and have h2d_ms = 0. For greedy, "rounds" = number of sequential removals.
// Verification (before any row is printed): every GPU run must equal cpu_batch_peel EXACTLY in best_num, best_den,
//   best_round, rounds AND the full removed_round array; the OpenMP result must equal the serial one. Any mismatch
//   or CUDA error => message on stderr, exit 1, no rows. On stderr also: density ratio gpu/greedy.
#include <cuda_runtime.h>
#include <string>
#include "densest.h"
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

// Allocates device memory, copies v to it and adds the (event-timed) copy time to *h2d_ms.
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

// Backend for batch_peel_loop (densest.h): owns the per-run device state, launches the kernels.
struct GpuBackend {
    uint32_t n, m;
    const uint32_t *rowptr, *col, *src;        // uploaded once, read-only
    uint8_t* alive = nullptr;                  // n bytes
    uint32_t* removed_round = nullptr;         // n
    uint32_t* tri = nullptr;                   // n, zeroed every round
    unsigned long long* stats = nullptr;       // stats[0] = T, stats[1] = removed count (one 16-byte copy back)
    int block = 256, grid_edges = 1, grid_vertices = 1;
    bool events = false;                       // diagnostics only (--verbose run): time the two kernels per round
    cudaEvent_t ev0, ev1;
    float last_ms = -1.f;

    void init(uint32_t n_, uint32_t m_, const uint32_t* rp, const uint32_t* cl, const uint32_t* sr) {
        n = n_; m = m_; rowptr = rp; col = cl; src = sr;
        CUDA_CHECK(cudaMalloc(&alive, std::max<size_t>(n, 1)));
        CUDA_CHECK(cudaMalloc(&removed_round, std::max<size_t>(n, 1) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&tri, std::max<size_t>(n, 1) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&stats, 2 * sizeof(unsigned long long)));
        auto blocks_for = [&](uint64_t threads) {
            uint64_t b = (threads + block - 1) / block;
            return (int)std::max<uint64_t>(1, std::min<uint64_t>(b, 1u << 20));
        };
        grid_edges = blocks_for(m);                // ~one thread per oriented edge (grid-stride loop covers the rest)
        grid_vertices = blocks_for(n);
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
    }
    void reset_state() {
        CUDA_CHECK(cudaMemset(alive, 1, n));                                  // every byte = 1: all vertices alive
        CUDA_CHECK(cudaMemset(removed_round, 0xFF, (size_t)n * sizeof(uint32_t)));   // UINT32_MAX = never removed
    }
    void begin_round() {                       // "reset_tri": tri[] and both counters back to zero
        CUDA_CHECK(cudaMemset(tri, 0, (size_t)n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(stats, 0, 2 * sizeof(unsigned long long)));
    }
    void run_tc() {
        if (events) CUDA_CHECK(cudaEventRecord(ev0));
        tc_per_vertex_alive<<<grid_edges, block>>>(rowptr, col, src, alive, m, tri, stats);
    }
    void run_mark(uint64_t nalive, uint64_t num, uint64_t den, uint32_t round) {
        mark_peel<<<grid_vertices, block>>>(tri, alive, removed_round, n, nalive, num, den, round, stats, stats + 1);
        if (events) CUDA_CHECK(cudaEventRecord(ev1));
    }
    void read_stats(uint64_t& T, uint64_t& removed) {   // blocking copy: waits for both kernels (the per-round sync)
        unsigned long long h[2];
        CUDA_CHECK(cudaMemcpy(h, stats, sizeof h, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());
        T = h[0]; removed = h[1];
        last_ms = -1.f;
        if (events) { CUDA_CHECK(cudaEventElapsedTime(&last_ms, ev0, ev1)); }
    }
    double kernel_ms() const { return last_ms; }
    void download_removed_round(std::vector<uint32_t>& out) {
        out.resize(n);
        if (n) CUDA_CHECK(cudaMemcpy(out.data(), removed_round, (size_t)n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
};

int main(int argc, char** argv) {
    std::vector<const char*> pos;
    bool burnin = true, greedy = true, verbose = false;
    int repeats = 5;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--pinned") == 0) g_pinned = true;
        else if (strcmp(argv[i], "--no-burnin") == 0) burnin = false;
        else if (strcmp(argv[i], "--no-greedy") == 0) greedy = false;
        else if (strcmp(argv[i], "--verbose") == 0) verbose = true;
        else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) repeats = std::max(1, atoi(argv[++i]));
        else pos.push_back(argv[i]);
    }
    if (pos.empty() || pos.size() == 2 || pos.size() > 3) {
        fprintf(stderr, "usage: %s graph.txt [eps_num eps_den] [--pinned] [--no-burnin] [--repeats N] [--no-greedy] [--verbose]\n", argv[0]);
        return 2;
    }
    const char* path = pos[0];
    const uint64_t num = pos.size() == 3 ? strtoull(pos[1], nullptr, 10) : 1;
    const uint64_t den = pos.size() == 3 ? strtoull(pos[2], nullptr, 10) : 10;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "GPU: %s, SMs=%d, cc=%d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    Graph g;
    if (!load_graph(path, g)) return 1;
    fprintf(stderr, "loaded n=%u m=%u max_out_deg=%u eps=%llu/%llu\n", g.n, g.m, g.max_out_degree,
            (unsigned long long)num, (unsigned long long)den);
    const uint64_t T_full = cpu_triangles_serial(g);

    // ---- CPU references (run first: they decide whether the GPU result is accepted) ----
    std::vector<std::string> rows;
    const std::string gpu_name = std::string("gpu_batch_peel") + (g_pinned ? "_pinned" : "");
    double t0 = now_ms();
    const PeelResult ref = cpu_batch_peel(g, num, den, false);
    const double t_ref = now_ms() - t0;
    if (!ref.ok) { fprintf(stderr, "cpu_batch_peel failed: %s\n", ref.error.c_str()); return 1; }
    rows.push_back(densest_row("cpu_batch_peel", path, g.n, g.m, ref.best_num, ref.best_den, ref.rounds, t_ref, 0));
#ifdef _OPENMP
    {
        t0 = now_ms();
        const PeelResult omp = cpu_batch_peel(g, num, den, true);
        const double t_omp = now_ms() - t0;
        if (!omp.ok || !(omp == ref)) { fprintf(stderr, "MISMATCH: OpenMP batch peel differs from serial\n"); return 1; }
        rows.push_back(densest_row("cpu_batch_peel_omp", path, g.n, g.m, omp.best_num, omp.best_den, omp.rounds, t_omp, 0));
    }
#endif
    GreedyResult gr;
    if (greedy) {
        t0 = now_ms();
        gr = cpu_greedy_peel(g);
        const double t_greedy = now_ms() - t0;
        if (!gr.ok) { fprintf(stderr, "cpu_greedy_peel failed: %s\n", gr.error.c_str()); return 1; }
        rows.push_back(densest_row("cpu_greedy_peel", path, g.n, g.m, gr.best_num, gr.best_den, g.n, t_greedy, 0));
    }

    // ---- GPU: upload once ----
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    double h2d_ms = 0;
    uint32_t* d_rowptr = upload(g.rowptr, &h2d_ms, e0, e1);
    uint32_t* d_col = upload(g.col, &h2d_ms, e0, e1);
    uint32_t* d_src = upload(g.src, &h2d_ms, e0, e1);
    GpuBackend be;
    be.init(g.n, g.m, d_rowptr, d_col, d_src);

    // Burn-in (untimed): keep the GPU busy ~1.5 s so the SM clock is ramped before ANY timed run (lesson from the
    // triangle-counting stage). Runs the new kernel on the all-alive graph; the state is reset again afterwards.
    if (burnin) {
        be.reset_state();
        const double tb = now_ms();
        do {
            be.begin_round();
            be.run_tc();
            CUDA_CHECK(cudaDeviceSynchronize());
        } while (now_ms() - tb < 1500.0);
        CUDA_CHECK(cudaGetLastError());
    }

    bool ok = true;
    // Runs the pipeline once, downloads removed_round (untimed) and checks EXACT equality with the CPU reference.
    auto run_once = [&](bool diag, double* ms_out) {
        be.events = diag;
        CUDA_CHECK(cudaDeviceSynchronize());
        const double ts = now_ms();
        PeelResult r = batch_peel_loop(g.n, T_full, num, den, be, diag, stderr);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (ms_out) *ms_out = now_ms() - ts;
        be.events = false;
        if (!r.ok) { fprintf(stderr, "GPU pipeline error: %s\n", r.error.c_str()); ok = false; return r; }
        be.download_removed_round(r.removed_round);
        if (!(r == ref)) {
            fprintf(stderr, "MISMATCH gpu vs cpu_batch_peel: best %llu/%llu round %u rounds %u (gpu) vs %llu/%llu round %u rounds %u (cpu); removed_round %s\n",
                    (unsigned long long)r.best_num, (unsigned long long)r.best_den, r.best_round, r.rounds,
                    (unsigned long long)ref.best_num, (unsigned long long)ref.best_den, ref.best_round, ref.rounds,
                    r.removed_round == ref.removed_round ? "equal" : "DIFFERS");
            ok = false;
        }
        return r;
    };
    run_once(false, nullptr);                                  // untimed warm-up + verification
    if (verbose) { fprintf(stderr, "--- per-round diagnostics (extra untimed run) ---\n"); run_once(true, nullptr); }
    double best_ms = 1e300;
    PeelResult last;
    for (int i = 0; i < repeats && ok; i++) {
        double ms = 0;
        last = run_once(false, &ms);
        best_ms = std::min(best_ms, ms);
    }
    if (!ok) { fprintf(stderr, "VERIFICATION FAILED - no timing rows printed\n"); return 1; }
    rows.push_back(densest_row(gpu_name.c_str(), path, g.n, g.m, last.best_num, last.best_den, last.rounds, best_ms, h2d_ms));

    for (auto& r : rows) puts(r.c_str());
    fflush(stdout);
    fprintf(stderr, "GPU result verified bit-exact against cpu_batch_peel (best %llu triangles / %llu vertices, %u rounds)\n",
            (unsigned long long)last.best_num, (unsigned long long)last.best_den, last.rounds);
    if (greedy) {
        const double dg = density_value(gr.best_num, gr.best_den);
        fprintf(stderr, "density ratio gpu/greedy = %.6f (gpu %.6f, greedy %.6f)\n",
                dg > 0 ? density_value(last.best_num, last.best_den) / dg : 0.0,
                density_value(last.best_num, last.best_den), dg);
    }
    return 0;
}
