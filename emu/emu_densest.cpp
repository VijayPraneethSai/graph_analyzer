// emu_densest.cpp - CPU-only tests of the triangle-density densest-subgraph pipeline. The CUDA kernels
// tc_per_vertex_alive and mark_peel (src/kernels.cuh) run through the thread emulator (cuda_shim.h), and the
// host loop is the REAL batch_peel_loop from src/densest.h with an emulator backend.
// usage:  emu_densest graph.txt     file mode: kernel tests + full pipeline on a (small) graph
//         emu_densest --tiny        random graphs with n <= 16: pipeline vs brute-force optimum over ALL subsets
// Independent oracles (not sharing code with the implementation under test): a brute-force adjacency-matrix
// triangle counter, a brute-force densest subset search, and an oracle that recomputes the best round from
// removed_round[] alone.
#include <random>
#include "cuda_shim.h"
#include "../src/graph.h"
#include "../src/densest.h"
#include "../src/kernels.cuh"

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

// ---- brute-force oracle: adjacency matrix built from the ORIGINAL oriented CSR edges ----
struct Brute {
    uint32_t n;
    std::vector<uint8_t> adj;   // n*n
    explicit Brute(const Graph& g) : n(g.n), adj((size_t)g.n * g.n, 0) {
        for (uint32_t e = 0; e < g.m; e++) { adj[(size_t)g.src[e] * n + g.col[e]] = 1; adj[(size_t)g.col[e] * n + g.src[e]] = 1; }
    }
    bool has(uint32_t a, uint32_t b) const { return adj[(size_t)a * n + b]; }
    // per-vertex triangles in the subgraph induced by alive; returns total. Loops over adjacent pairs u<v and w>v.
    uint64_t participation(const std::vector<uint8_t>& alive, std::vector<uint32_t>& tri) const {
        tri.assign(n, 0);
        uint64_t T = 0;
        for (uint32_t u = 0; u < n; u++) {
            if (!alive[u]) continue;
            for (uint32_t v = u + 1; v < n; v++) {
                if (!alive[v] || !has(u, v)) continue;
                for (uint32_t w = v + 1; w < n; w++)
                    if (alive[w] && has(u, w) && has(v, w)) { T++; tri[u]++; tri[v]++; tri[w]++; }
            }
        }
        return T;
    }
};

// Oracle for the best round using only removed_round[] and brute force: first round (strictly better) whose
// alive set {removed_round >= r} has the maximum density. Returns false if there are no rounds.
static void oracle_best(const Brute& b, const std::vector<uint32_t>& rr, uint32_t rounds, uint64_t& bn, uint64_t& bd, uint32_t& br) {
    bn = 0; bd = 0; br = 0;
    std::vector<uint32_t> tri;
    for (uint32_t r = 0; r < rounds; r++) {
        std::vector<uint8_t> alive(b.n);
        uint64_t cnt = 0;
        for (uint32_t v = 0; v < b.n; v++) { alive[v] = rr[v] >= r; cnt += alive[v]; }
        uint64_t T = b.participation(alive, tri);
        if (bd == 0 || T * bd > bn * cnt) { bn = T; bd = cnt; br = r; }
    }
}

// ---- emulator backend for batch_peel_loop ----
struct EmuBackend {
    const Graph& g;
    unsigned grid, block;
    std::vector<uint8_t> alive;
    std::vector<uint32_t> rr, tri;
    unsigned long long stats[2] = {0, 0};
    EmuBackend(const Graph& g_, unsigned grid_, unsigned block_) : g(g_), grid(grid_), block(block_) {}
    void reset_state() { alive.assign(g.n, 1); rr.assign(g.n, UINT32_MAX); }
    void begin_round() { tri.assign(g.n, 0); stats[0] = stats[1] = 0; }
    void run_tc() {
        emu_launch(grid, block, [&] { tc_per_vertex_alive(g.rowptr.data(), g.col.data(), g.src.data(), alive.data(), g.m, tri.data(), &stats[0]); });
    }
    void run_mark(uint64_t nalive, uint64_t num, uint64_t den, uint32_t round) {
        emu_launch(grid, block, [&] { mark_peel(tri.data(), alive.data(), rr.data(), g.n, nalive, num, den, round, &stats[0], &stats[1]); });
    }
    void read_stats(uint64_t& T, uint64_t& removed) { T = stats[0]; removed = stats[1]; }
    double kernel_ms() const { return -1; }
};

static PeelResult emu_pipeline(const Graph& g, uint64_t T_full, uint64_t num, uint64_t den, unsigned grid, unsigned block) {
    EmuBackend be(g, grid, block);
    PeelResult r = batch_peel_loop(g.n, T_full, num, den, be);
    r.removed_round = be.rr;
    return r;
}

// ---- brute-force densest subset (n <= 16): max over all non-empty subsets of triangles/|S| ----
static void brute_densest(const Brute& b, uint64_t& bn, uint64_t& bd) {
    std::vector<uint32_t> tmask;
    for (uint32_t u = 0; u < b.n; u++) for (uint32_t v = u + 1; v < b.n; v++) if (b.has(u, v))
        for (uint32_t w = v + 1; w < b.n; w++) if (b.has(u, w) && b.has(v, w)) tmask.push_back((1u << u) | (1u << v) | (1u << w));
    bn = 0; bd = 1;
    for (uint32_t S = 1; S < (1u << b.n); S++) {
        uint64_t t = 0;
        for (uint32_t tm : tmask) t += (S & tm) == tm;
        uint64_t sz = __builtin_popcount(S);
        if (t * bd > bn * sz) { bn = t; bd = sz; }
    }
}

static void test_kernels(const Graph& g, const Brute& b, std::mt19937& rng, uint64_t T_full) {
    const double probs[] = {1.0, 0.9, 0.6, 0.3, 0.05, 0.0};
    struct Cfg { unsigned grid, block; } cfgs[] = {{1, 32}, {3, 64}};
    for (double p : probs) {
        std::vector<uint8_t> alive(g.n);
        for (auto& a : alive) a = std::generate_canonical<double, 32>(rng) < p;
        std::vector<uint32_t> want;
        uint64_t want_T = b.participation(alive, want);
        if (p == 1.0) CHECK(want_T == T_full, "brute full count %llu != serial %llu", (unsigned long long)want_T, (unsigned long long)T_full);
        for (auto c : cfgs) {
            std::vector<uint32_t> tri(g.n, 0);
            unsigned long long total = 0;
            emu_launch(c.grid, c.block, [&] { tc_per_vertex_alive(g.rowptr.data(), g.col.data(), g.src.data(), alive.data(), g.m, tri.data(), &total); });
            CHECK(total == want_T, "tc_per_vertex_alive total p=%.2f grid=%u block=%u: %llu vs %llu", p, c.grid, c.block, total, (unsigned long long)want_T);
            CHECK(tri == want, "tc_per_vertex_alive tri[] differs p=%.2f grid=%u block=%u", p, c.grid, c.block);
            std::vector<uint32_t> cpu;
            CHECK(cpu_triangle_participation(g, alive, cpu) == want_T && cpu == want, "cpu_triangle_participation differs p=%.2f", p);
            CHECK(cpu_triangle_participation_omp(g, alive, cpu) == want_T && cpu == want, "cpu_triangle_participation_omp differs p=%.2f", p);

            // mark_peel on the true tri[]: compare with an independent 128-bit evaluation of the threshold
            const uint64_t eps[][2] = {{1, 10}, {1, 1}, {0, 1}, {3, 7}};
            for (auto& e : eps) {
                uint64_t nalive = 0;
                for (uint32_t v = 0; v < g.n; v++) nalive += alive[v];
                if (nalive == 0) continue;
                std::vector<uint8_t> a2 = alive;
                std::vector<uint32_t> rr(g.n, UINT32_MAX);
                unsigned long long tot = want_T, removed = 0;
                const uint32_t round = 5;
                emu_launch(c.grid, c.block, [&] { mark_peel(want.data(), a2.data(), rr.data(), g.n, nalive, e[0], e[1], round, &tot, &removed); });
                unsigned long long want_removed = 0;
                for (uint32_t v = 0; v < g.n; v++) {
                    bool rem = alive[v] && (unsigned __int128)want[v] * nalive * e[1] <= (unsigned __int128)3 * (e[1] + e[0]) * want_T;
                    CHECK(a2[v] == (alive[v] && !rem), "mark_peel alive[%u] wrong (eps %llu/%llu)", v, (unsigned long long)e[0], (unsigned long long)e[1]);
                    CHECK(rr[v] == (rem ? round : UINT32_MAX), "mark_peel removed_round[%u] wrong", v);
                    want_removed += rem;
                }
                CHECK(removed == want_removed, "mark_peel removed count %llu vs %llu", removed, want_removed);
                if (g_fail > 20) return;
            }
        }
    }
}

// full pipeline on one graph: emulated GPU loop vs cpu_batch_peel (exact), plus independent oracles
static void test_pipeline(const Graph& g, const Brute& b, uint64_t T_full, uint64_t num, uint64_t den, unsigned grid, unsigned block, bool oracle) {
    PeelResult ref = cpu_batch_peel(g, num, den, false);
    PeelResult omp = cpu_batch_peel(g, num, den, true);
    PeelResult emu = emu_pipeline(g, T_full, num, den, grid, block);
    CHECK(ref.ok, "cpu_batch_peel failed: %s", ref.error.c_str());
    CHECK(emu.ok, "emulated pipeline failed: %s", emu.error.c_str());
    CHECK(omp == ref, "cpu_batch_peel_omp != cpu_batch_peel");
    CHECK(emu == ref, "emulated GPU pipeline != cpu_batch_peel (best %llu/%llu round %u rounds %u vs %llu/%llu round %u rounds %u)",
          (unsigned long long)emu.best_num, (unsigned long long)emu.best_den, emu.best_round, emu.rounds,
          (unsigned long long)ref.best_num, (unsigned long long)ref.best_den, ref.best_round, ref.rounds);
    if (!oracle || !ref.ok) return;
    for (uint32_t v = 0; v < g.n; v++) CHECK(ref.removed_round[v] < ref.rounds, "vertex %u never removed", v);
    // (1) best subgraph recomputed from removed_round[] must have the claimed triangles/vertices
    std::vector<uint8_t> alive(g.n);
    uint64_t cnt = 0;
    for (uint32_t v = 0; v < g.n; v++) { alive[v] = ref.removed_round[v] >= ref.best_round; cnt += alive[v]; }
    std::vector<uint32_t> tri;
    uint64_t T = b.participation(alive, tri);
    CHECK(T == ref.best_num && cnt == ref.best_den, "best subgraph from removed_round: %llu/%llu, claimed %llu/%llu",
          (unsigned long long)T, (unsigned long long)cnt, (unsigned long long)ref.best_num, (unsigned long long)ref.best_den);
    // (2) best_round must be the FIRST round of maximum density
    uint64_t on, od; uint32_t orr;
    oracle_best(b, ref.removed_round, ref.rounds, on, od, orr);
    CHECK(on == ref.best_num && od == ref.best_den && orr == ref.best_round, "oracle best round %u (%llu/%llu) vs %u (%llu/%llu)",
          orr, (unsigned long long)on, (unsigned long long)od, ref.best_round, (unsigned long long)ref.best_num, (unsigned long long)ref.best_den);
}

static int run_file(const char* path) {
    Graph g;
    if (!load_graph(path, g)) return 1;
    const uint64_t T_full = cpu_triangles_serial(g);
    Brute b(g);
    std::mt19937 rng(12345);
    printf("file mode: n=%u m=%u triangles=%llu\n", g.n, g.m, (unsigned long long)T_full);
    test_kernels(g, b, rng, T_full);
    const uint64_t eps[][2] = {{1, 10}, {1, 1}, {0, 1}};
    for (auto& e : eps) {
        test_pipeline(g, b, T_full, e[0], e[1], 3, 64, true);
        PeelResult ref = cpu_batch_peel(g, e[0], e[1]);
        GreedyResult gr = cpu_greedy_peel(g);
        CHECK(gr.ok, "greedy failed: %s", gr.error.c_str());
        CHECK(gr.best_den >= 1 && gr.best_num * g.n >= T_full * gr.best_den, "greedy must be at least as dense as the full graph");
        printf("  eps=%llu/%llu  batch best=%llu/%llu (%.4f) rounds=%u   greedy best=%llu/%llu (%.4f)\n", (unsigned long long)e[0], (unsigned long long)e[1],
               (unsigned long long)ref.best_num, (unsigned long long)ref.best_den, density_value(ref.best_num, ref.best_den), ref.rounds,
               (unsigned long long)gr.best_num, (unsigned long long)gr.best_den, density_value(gr.best_num, gr.best_den));
    }
    return 0;
}

static int run_tiny() {
    std::mt19937 rng(777);
    const double ps[] = {0.15, 0.3, 0.5, 0.7, 0.9};
    const uint64_t eps[][2] = {{1, 10}, {1, 2}, {1, 1}, {3, 1}, {0, 1}};
    int graphs = 0, with_tri = 0;
    for (int iter = 0; iter < 60; iter++) {
        uint32_t n = 6 + rng() % 11;                       // 6..16
        double p = ps[iter % 5];
        std::vector<std::pair<uint32_t, uint32_t>> edges;
        auto add = [&](uint32_t u, uint32_t v) { if (u != v) edges.emplace_back(std::min(u, v), std::max(u, v)); };
        for (uint32_t u = 0; u < n; u++) for (uint32_t v = u + 1; v < n; v++) if (std::generate_canonical<double, 32>(rng) < p) add(u, v);
        if (iter % 4 == 0) { uint32_t k = 4 + rng() % 3; for (uint32_t u = 0; u < k; u++) for (uint32_t v = u + 1; v < k; v++) add(u, v); }   // planted clique
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        if (edges.empty()) continue;
        FILE* f = fopen("bin/_tiny_graph.txt", "w");
        for (auto& e : edges) fprintf(f, "%u %u\n", e.first, e.second);
        fclose(f);
        Graph g;
        if (!load_graph("bin/_tiny_graph.txt", g)) return 1;
        Brute b(g);
        uint64_t T_full = cpu_triangles_serial(g);
        uint64_t on, od;
        brute_densest(b, on, od);
        graphs++; with_tri += T_full > 0;
        for (auto& e : eps) {
            const uint64_t num = e[0], den = e[1];
            test_pipeline(g, b, T_full, num, den, 2, 32, true);
            PeelResult r = cpu_batch_peel(g, num, den);
            // batch result is a real subset => cannot beat the optimum
            CHECK(r.best_num * od <= on * r.best_den, "batch density exceeds brute-force optimum (iter %d)", iter);
            // approximation guarantee: opt <= 3(1+eps) * found   <=>   on*fd*den <= 3(den+num) * fn * od
            CHECK(on * r.best_den * den <= 3 * (den + num) * r.best_num * od, "GUARANTEE violated iter %d eps %llu/%llu: opt %llu/%llu found %llu/%llu",
                  iter, (unsigned long long)num, (unsigned long long)den, (unsigned long long)on, (unsigned long long)od,
                  (unsigned long long)r.best_num, (unsigned long long)r.best_den);
        }
        GreedyResult gr = cpu_greedy_peel(g);
        CHECK(gr.ok, "greedy failed: %s", gr.error.c_str());
        CHECK(gr.best_den >= 1 && gr.best_den <= g.n, "greedy recorded no subgraph (iter %d)", iter);
        CHECK(gr.best_num * od <= on * gr.best_den, "greedy density exceeds optimum (iter %d)", iter);
        CHECK(gr.best_num * g.n >= T_full * gr.best_den, "greedy worse than the full graph it started from (iter %d)", iter);
        CHECK(on * gr.best_den <= 3 * gr.best_num * od, "greedy 3-approximation violated iter %d: opt %llu/%llu greedy %llu/%llu", iter,
              (unsigned long long)on, (unsigned long long)od, (unsigned long long)gr.best_num, (unsigned long long)gr.best_den);
        if (g_fail > 20) break;
    }
    printf("tiny mode: %d graphs (%d with triangles), 5 eps values each, all subsets brute-forced\n", graphs, with_tri);
    CHECK(with_tri >= 30, "too few tiny graphs with triangles to be a meaningful test");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    int rc = strcmp(argv[1], "--tiny") == 0 ? run_tiny() : run_file(argv[1]);
    if (rc) return rc;
    printf(g_fail ? "FAILED (%d checks)\n" : "ALL OK\n", g_fail);
    return g_fail ? 1 : 0;
}
