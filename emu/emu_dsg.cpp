// emu_dsg.cpp - CPU-only tests of the stage-6 paper algorithms (src/dsg.h) and of the GPU core-decomposition kernels
// (core_min_alive_tri, core_peel_level, tc_per_vertex_alive from src/kernels.cuh), which run through the thread emulator
// (cuda_shim.h) inside the REAL host loop core_decomp_loop from src/dsg.h. Nothing here touches a GPU.
// usage:  emu_dsg --tiny        many small random + hand-made graphs (n <= 12), everything compared with BRUTE FORCE over
//                               all 2^n vertex subsets
//         emu_dsg --mid         160 medium random graphs with planted communities: Exact is the oracle for CoreExact
//         emu_dsg graph.txt     larger graph (n up to ~1000): no brute force; independent cross-checks instead
// Oracles that share no code with the implementation under test:
//   * Brute: adjacency matrix + list of all triangle bitmasks; for every vertex subset S the exact triangle degrees in S.
//     rho_opt = max_S triangles(S)/|S| (exact fractions), core[v] = max over subsets S containing v of min_{u in S} tri_S(u)
//     = the DEFINITION of the (k,Psi)-core number ("largest subgraph with min triangle degree >= k", unions of valid subsets
//     are valid, so the maximum over subsets is the core number).
//   * core_by_definition (file mode): for k = 0,1,...: repeatedly delete vertices with fewer than k triangles until stable.
//   * cpu_greedy_peel (stage 5 code) as an independent Charikar greedy.
// Checks per graph (tiny mode): see test_graph(). Printed at the end: how many graphs, how often Greedy++ reached the optimum.
#include <random>
#include "cuda_shim.h"
#include "../src/graph.h"
#include "../src/dsg.h"
#include "../src/kernels.cuh"

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; if (g_fail <= 25) { printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

typedef std::vector<std::pair<uint32_t, uint32_t>> EdgeList;

// ---- brute-force oracle (n <= 12) ----
struct Brute {
    uint32_t n;
    std::vector<uint8_t> adj;
    std::vector<uint32_t> tmask;            // one bitmask per triangle
    explicit Brute(const Graph& g) : n(g.n), adj((size_t)g.n * g.n, 0) {
        for (uint32_t e = 0; e < g.m; e++) { adj[(size_t)g.src[e] * n + g.col[e]] = 1; adj[(size_t)g.col[e] * n + g.src[e]] = 1; }
        for (uint32_t u = 0; u < n; u++) for (uint32_t v = u + 1; v < n; v++) if (has(u, v))
            for (uint32_t w = v + 1; w < n; w++) if (has(u, w) && has(v, w)) tmask.push_back((1u << u) | (1u << v) | (1u << w));
    }
    bool has(uint32_t a, uint32_t b) const { return adj[(size_t)a * n + b]; }
    uint64_t tri_in(uint32_t S) const { uint64_t t = 0; for (uint32_t tm : tmask) t += (S & tm) == tm; return t; }
    // per-vertex triangle degrees inside S
    void degrees(uint32_t S, std::vector<uint32_t>& d) const {
        d.assign(n, 0);
        for (uint32_t tm : tmask) if ((S & tm) == tm) for (uint32_t v = 0; v < n; v++) if (tm >> v & 1) d[v]++;
    }
    // optimum density and brute-force core numbers over ALL non-empty subsets
    void solve(Dens& opt, std::vector<uint32_t>& core) const {
        opt = Dens{0, 0};
        core.assign(n, 0);
        std::vector<uint32_t> d;
        for (uint32_t S = 1; S < (1u << n); S++) {
            degrees(S, d);
            uint32_t mind = UINT32_MAX;
            uint64_t t = 0, sz = 0;
            for (uint32_t v = 0; v < n; v++) if (S >> v & 1) { mind = std::min(mind, d[v]); t += d[v]; sz++; }
            t /= 3;
            if (dens_gt(Dens{t, sz}, opt)) opt = Dens{t, sz};
            for (uint32_t v = 0; v < n; v++) if (S >> v & 1) core[v] = std::max(core[v], mind);
        }
    }
    Dens recount(const std::vector<uint32_t>& set) const {
        uint32_t S = 0;
        for (uint32_t v : set) { CHECK(v < n && !(S >> v & 1), "bad/duplicate vertex %u in returned set", v); S |= 1u << v; }
        return Dens{tri_in(S), set.size()};
    }
};

// ---- definitional core numbers for larger graphs: k-core = fixpoint of deleting vertices with < k triangles ----
static std::vector<uint32_t> core_by_definition(const Graph& g) {
    std::vector<uint32_t> core(g.n, 0), tri;
    std::vector<uint8_t> alive(g.n, 1);
    for (uint32_t k = 1;; k++) {
        for (;;) {                                    // delete until stable: the k-core
            cpu_triangle_participation(g, alive, tri);
            bool changed = false;
            for (uint32_t v = 0; v < g.n; v++) if (alive[v] && tri[v] < k) { alive[v] = 0; changed = true; }
            if (!changed) break;
        }
        bool any = false;
        for (uint32_t v = 0; v < g.n; v++) if (alive[v]) { core[v] = k; any = true; }
        if (!any) break;
    }
    return core;
}

// ---- emulated GPU backend for core_decomp_loop ----
struct EmuCoreBackend {
    const Graph& g;
    unsigned grid, block;
    std::vector<uint8_t> alive;
    std::vector<uint32_t> core, tri;
    unsigned min_tri = 0xFFFFFFFFu;
    unsigned long long stats[3] = {0, 0, 0};
    EmuCoreBackend(const Graph& g_, unsigned grid_, unsigned block_) : g(g_), grid(grid_), block(block_) {}
    void reset_state() { alive.assign(g.n, 1); core.assign(g.n, UINT32_MAX); }
    void begin_round() { tri.assign(g.n, 0); stats[0] = stats[1] = stats[2] = 0; min_tri = 0xFFFFFFFFu; }
    void run_tc() { emu_launch(grid, block, [&] { tc_per_vertex_alive(g.rowptr.data(), g.col.data(), g.src.data(), alive.data(), g.m, tri.data(), &stats[0]); }); }
    void run_min() { emu_launch(grid, block, [&] { core_min_alive_tri(tri.data(), alive.data(), g.n, &min_tri); }); }
    void run_peel(uint64_t k) { emu_launch(grid, block, [&] { core_peel_level(tri.data(), alive.data(), core.data(), g.n, k, &min_tri, &stats[1], &stats[2]); }); }
    void read_stats(uint64_t& T, uint64_t& removed, uint64_t& k) { T = stats[0]; removed = stats[1]; k = stats[2]; }
};


// ---- emulated GPU backend for core_decomp_loop_inc (incremental frontier kernels) ----
struct EmuCoreBackendInc {
    const Graph& g;
    unsigned grid, block;
    std::vector<uint32_t> urow, ucol;
    std::vector<uint8_t> state;
    std::vector<uint32_t> core, tri, frontier;
    unsigned min_tri = 0xFFFFFFFFu;
    unsigned long long stats[3] = {0, 0, 0};          // [0] = T (init), [1] = frontier size this round, [2] = level k
    EmuCoreBackendInc(const Graph& g_, unsigned grid_, unsigned block_) : g(g_), grid(grid_), block(block_) { build_undirected_csr(g, urow, ucol); }
    void reset_state() { state.assign(g.n, 1); core.assign(g.n, UINT32_MAX); frontier.assign(std::max<uint32_t>(g.n, 1), 0); }
    void init_tri() {
        tri.assign(g.n, 0); stats[0] = 0;
        emu_launch(grid, block, [&] { tc_per_vertex_alive(g.rowptr.data(), g.col.data(), g.src.data(), state.data(), g.m, tri.data(), &stats[0]); });
    }
    void init_tri(uint64_t& T) { init_tri(); T = stats[0]; }
    void begin_round() { stats[1] = stats[2] = 0; min_tri = 0xFFFFFFFFu; }
    void run_min() { emu_launch(grid, block, [&] { core_min_alive_tri(tri.data(), state.data(), g.n, &min_tri); }); }
    void run_find(uint64_t k) { emu_launch(grid, block, [&] { core_find_frontier(tri.data(), state.data(), core.data(), g.n, k, &min_tri, frontier.data(), &stats[1], &stats[2]); }); }
    void run_update() { emu_launch(grid, block, [&] { core_update_frontier(urow.data(), ucol.data(), state.data(), frontier.data(), &stats[1], tri.data()); }); }
    void run_retire() { emu_launch(grid, block, [&] { core_retire_frontier(state.data(), frontier.data(), &stats[1]); }); }
    void read_stats(uint64_t& removed, uint64_t& k) { removed = stats[1]; k = stats[2]; }
};

static std::mt19937 rng(20261010);
static double urand() { return std::generate_canonical<double, 32>(rng); }

// ---- structural validation of a Graph against an expected UNDIRECTED edge set (independent of build_graph_from_edges) ----
static void validate_graph(const Graph& g, uint32_t n, EdgeList expect, const char* what) {
    for (auto& e : expect) if (e.first > e.second) std::swap(e.first, e.second);
    std::sort(expect.begin(), expect.end());
    CHECK(g.n == n, "%s: n %u vs %u", what, g.n, n);
    CHECK(g.m == expect.size(), "%s: m %u vs %zu", what, g.m, expect.size());
    if (g.n != n || g.m != expect.size()) return;
    CHECK(g.rowptr.size() == n + 1u && g.col.size() == g.m && g.src.size() == g.m, "%s: array sizes", what);
    if (g.rowptr.size() != n + 1u || g.col.size() != g.m || g.src.size() != g.m) return;
    std::vector<uint32_t> deg(n, 0);
    for (auto& e : expect) { deg[e.first]++; deg[e.second]++; }
    CHECK(g.rowptr[0] == 0 && g.rowptr[n] == g.m, "%s: rowptr ends", what);
    EdgeList got;
    uint32_t maxod = 0;
    for (uint32_t u = 0; u < n; u++) {
        CHECK(g.rowptr[u] <= g.rowptr[u + 1], "%s: rowptr not monotone at %u", what, u);
        if (g.rowptr[u] > g.rowptr[u + 1] || g.rowptr[u + 1] > g.m) return;
        maxod = std::max(maxod, g.rowptr[u + 1] - g.rowptr[u]);
        for (uint32_t e = g.rowptr[u]; e < g.rowptr[u + 1]; e++) {
            const uint32_t v = g.col[e];
            CHECK(g.src[e] == u, "%s: src[%u] = %u, row owner %u", what, e, g.src[e], u);
            CHECK(e == g.rowptr[u] || g.col[e - 1] < v, "%s: row %u not strictly sorted", what, u);
            CHECK(v < n && u != v, "%s: bad endpoint", what);
            const bool fwd = deg[u] != deg[v] ? deg[u] < deg[v] : u < v;       // orientation rule: smaller (degree, id) first
            CHECK(fwd, "%s: edge %u->%u violates the (degree,id) orientation", what, u, v);
            got.emplace_back(std::min(u, v), std::max(u, v));
        }
    }
    std::sort(got.begin(), got.end());
    CHECK(got == expect, "%s: edge set differs", what);
    CHECK(g.max_out_degree == maxod, "%s: max_out_degree %u vs %u", what, g.max_out_degree, maxod);
}

static bool graphs_equal(const Graph& a, const Graph& b) {
    return a.n == b.n && a.m == b.m && a.rowptr == b.rowptr && a.col == b.col && a.src == b.src && a.max_out_degree == b.max_out_degree;
}

// ---- kernel-level tests of core_min_alive_tri / core_peel_level on random alive masks with the TRUE tri[] ----
static void test_core_kernels(const Graph& g, const Brute& b) {
    const double probs[] = {1.0, 0.7, 0.3, 0.0};
    struct Cfg { unsigned grid, block; } cfgs[] = {{1, 32}, {3, 64}};
    for (double p : probs) {
        uint32_t S = 0;
        std::vector<uint8_t> alive(g.n);
        for (uint32_t v = 0; v < g.n; v++) { alive[v] = urand() < p; if (alive[v]) S |= 1u << v; }
        std::vector<uint32_t> d;
        b.degrees(S, d);
        unsigned want_min = 0xFFFFFFFFu;
        for (uint32_t v = 0; v < g.n; v++) if (alive[v]) want_min = std::min(want_min, d[v]);
        for (auto c : cfgs) {
            unsigned mn = 0xFFFFFFFFu;
            emu_launch(c.grid, c.block, [&] { core_min_alive_tri(d.data(), alive.data(), g.n, &mn); });
            CHECK(mn == want_min, "core_min_alive_tri: %u vs %u (p=%.1f grid=%u block=%u)", mn, want_min, p, c.grid, c.block);
            if (!S) continue;
            for (uint64_t k_in : {0ull, 1ull, 2ull, 5ull}) {
                std::vector<uint8_t> a2 = alive;
                std::vector<uint32_t> core(g.n, UINT32_MAX);
                unsigned long long removed = 0, k_out = 7777;
                unsigned mn2 = want_min;
                emu_launch(c.grid, c.block, [&] { core_peel_level(d.data(), a2.data(), core.data(), g.n, k_in, &mn2, &removed, &k_out); });
                const uint64_t k = std::max<uint64_t>(k_in, want_min);
                unsigned long long want_removed = 0;
                for (uint32_t v = 0; v < g.n; v++) {
                    const bool rem = alive[v] && d[v] <= k;
                    want_removed += rem;
                    CHECK(a2[v] == (alive[v] && !rem), "core_peel_level alive[%u] wrong", v);
                    CHECK(core[v] == (rem ? (uint32_t)k : UINT32_MAX), "core_peel_level core[%u] wrong", v);
                }
                CHECK(removed == want_removed, "core_peel_level removed %llu vs %llu", removed, want_removed);
                CHECK(k_out == k, "core_peel_level k_out %llu vs %llu", k_out, (unsigned long long)k);
            }
        }
    }
}

struct Stats {
    int graphs = 0, with_tri = 0, no_tri = 0;
    int greedy_gap = 0;           // graphs where Charikar greedy is strictly below the optimum
    int pp_reached = 0;           // ... and Greedy++ with Tmax passes reached the optimum
    int pp_all_reached = 0;       // graphs (with triangles) where Greedy++ reached the optimum within Tmax passes
    int pp_better_than_greedy = 0;
    int first_T_hist[8] = {0};    // for graphs where it reached: bucket of the first pass that hit the optimum
    int prune_nontrivial = 0;     // graphs where the pruned graph is strictly smaller than the input
    long long seeded_flows = 0, unseeded_flows = 0;   // total max-flow calls of the tiny suite with / without a Greedy++ seed
    int weak_shrinks = 0, weak_multi = 0;   // graphs where the weak-start CoreExact used the line-17 shrink / searched >1 component
} g_stats;

static const uint32_t TMAX = 30;

// Everything checked on one small graph against brute force.
static void test_graph(const Graph& g, const Brute& b, const EdgeList& edges, int idx, bool heavy_emu) {
    char tag[64];
    snprintf(tag, sizeof tag, "graph #%d (n=%u m=%u)", idx, g.n, g.m);
    g_stats.graphs++;
    Dens opt; std::vector<uint32_t> bcore;
    b.solve(opt, bcore);
    const uint64_t T_full = cpu_triangles_serial(g);
    CHECK(T_full == b.tmask.size(), "%s: serial triangle count %llu vs brute %zu", tag, (unsigned long long)T_full, b.tmask.size());
    (opt.num ? g_stats.with_tri : g_stats.no_tri)++;
    validate_graph(g, g.n, edges, tag);

    // 1. core decomposition (Alg 3) vs the definition (brute force) and vs the fixpoint implementation
    std::vector<uint32_t> core; uint32_t kmax = 0;
    CHECK(core_decomposition(g, core, &kmax), "%s: core_decomposition failed", tag);
    CHECK(core == bcore, "%s: Alg 3 core numbers differ from the brute-force definition", tag);
    CHECK(core == core_by_definition(g), "%s: Alg 3 differs from the fixpoint definition", tag);
    CHECK(kmax == (g.n ? *std::max_element(bcore.begin(), bcore.end()) : 0), "%s: kmax", tag);
    { std::vector<uint64_t> td; triangle_degrees(g, td); for (uint32_t v = 0; v < g.n; v++) CHECK(core[v] <= td[v], "%s: core[%u] exceeds its triangle degree", tag, v); }

    // 2. emulated GPU level-synchronous decomposition == CPU Alg 3
    {
        struct Cfg { unsigned grid, block; } cfgs[] = {{2, 32}, {3, 64}};
        for (int ci = 0; ci < (heavy_emu ? 2 : 1); ci++) {
            EmuCoreBackend be(g, cfgs[ci].grid, cfgs[ci].block);
            CoreLoopResult r = core_decomp_loop(g.n, T_full, be);
            CHECK(r.ok, "%s: emulated core loop failed: %s", tag, r.error.c_str());
            CHECK(be.core == core, "%s: emulated GPU core numbers differ from CPU Alg 3 (grid %u block %u)", tag, cfgs[ci].grid, cfgs[ci].block);
            CHECK(r.kmax == kmax, "%s: emulated kmax %u vs %u", tag, r.kmax, kmax);
            CHECK(r.rounds >= 1 && r.rounds <= g.n, "%s: rounds %u", tag, r.rounds);
        }
        if (heavy_emu) test_core_kernels(g, b);
    }
    // 2b. emulated INCREMENTAL frontier decomposition == CPU Alg 3 (several grid/block shapes: the warp-per-vertex update kernel
    //     must not depend on how many warps are launched). Also: tri[] of every still-alive vertex stays equal to the true count.
    {
        struct Cfg { unsigned grid, block; } cfgs[] = {{1, 32}, {2, 32}, {3, 64}};
        for (int ci = 0; ci < (heavy_emu ? 3 : 2); ci++) {
            EmuCoreBackendInc be(g, cfgs[ci].grid, cfgs[ci].block);
            CoreLoopResult r = core_decomp_loop_inc(g.n, T_full, be);
            CHECK(r.ok, "%s: emulated incremental core loop failed: %s", tag, r.error.c_str());
            CHECK(be.core == core, "%s: emulated INCREMENTAL core numbers differ from CPU Alg 3 (grid %u block %u)", tag, cfgs[ci].grid, cfgs[ci].block);
            CHECK(r.kmax == kmax, "%s: incremental kmax %u vs %u", tag, r.kmax, kmax);
            CHECK(r.rounds >= 1 && r.rounds <= g.n, "%s: incremental rounds %u", tag, r.rounds);
        }
        // the number of rounds must equal the recount loop's (same level rule, same removals)
        EmuCoreBackend b1(g, 2, 32); EmuCoreBackendInc b2(g, 2, 32);
        CoreLoopResult r1 = core_decomp_loop(g.n, T_full, b1), r2 = core_decomp_loop_inc(g.n, T_full, b2);
        CHECK(r1.ok && r2.ok && r1.rounds == r2.rounds, "%s: recount rounds %u vs incremental rounds %u", tag, r1.rounds, r2.rounds);
    }

    // 3. Exact (Alg 1) and CoreExact (Alg 4) == brute-force optimum, with a witness set
    DsgResult ex = exact_densest(g);
    CHECK(ex.ok, "%s: exact_densest: %s", tag, ex.error.c_str());
    CHECK(dens_cmp(ex.dens(), opt) == 0, "%s: exact_densest %llu/%llu vs brute %llu/%llu", tag, (unsigned long long)ex.tri, (unsigned long long)ex.verts, (unsigned long long)opt.num, (unsigned long long)opt.den);
    CHECK(ex.set.size() == ex.verts, "%s: exact set size", tag);
    { Dens rc = b.recount(ex.set); CHECK(ex.verts == 0 ? rc.den == 0 : dens_cmp(rc, ex.dens()) == 0 && rc.num == ex.tri, "%s: exact witness recount", tag); }
    DsgResult ce = core_exact(g, core);
    CHECK(ce.ok, "%s: core_exact: %s", tag, ce.error.c_str());
    CHECK(dens_cmp(ce.dens(), opt) == 0, "%s: core_exact %llu/%llu vs brute %llu/%llu", tag, (unsigned long long)ce.tri, (unsigned long long)ce.verts, (unsigned long long)opt.num, (unsigned long long)opt.den);
    CHECK(ce.set.size() == ce.verts, "%s: core_exact set size", tag);
    { Dens rc = b.recount(ce.set); CHECK(ce.verts == 0 ? rc.den == 0 : dens_cmp(rc, ce.dens()) == 0 && rc.num == ce.tri, "%s: core_exact witness recount", tag); }
    {   // weak start (no Pruning1/2, one triangle as the first lower bound): the binary search itself must find the optimum
        DsgResult cw = core_exact(g, core, false);
        CHECK(cw.ok && dens_cmp(cw.dens(), opt) == 0, "%s: core_exact(weak start) %llu/%llu vs brute %llu/%llu (%s)", tag, (unsigned long long)cw.tri, (unsigned long long)cw.verts, (unsigned long long)opt.num, (unsigned long long)opt.den, cw.error.c_str());
        Dens rc = b.recount(cw.set);
        CHECK(cw.verts == 0 ? rc.den == 0 : dens_cmp(rc, cw.dens()) == 0 && rc.num == cw.tri, "%s: core_exact(weak start) witness recount", tag);
        g_stats.weak_shrinks += cw.shrinks > 0;
        g_stats.weak_multi += cw.components_searched > 1;
    }

    {   // seeded CoreExact: a seed from Greedy (always) must not change the optimum; a bogus seed must be rejected
        DsgResult sg = greedy(g);
        DsgResult cs = core_exact(g, core, true, &sg);
        CHECK(cs.ok && dens_cmp(cs.dens(), opt) == 0, "%s: core_exact(seed=greedy) %llu/%llu vs brute %llu/%llu (%s)", tag, (unsigned long long)cs.tri, (unsigned long long)cs.verts, (unsigned long long)opt.num, (unsigned long long)opt.den, cs.error.c_str());
        DsgResult sp = greedy_pp(g, 8);
        DsgResult cp2 = core_exact(g, core, true, &sp);
        CHECK(cp2.ok && dens_cmp(cp2.dens(), opt) == 0, "%s: core_exact(seed=greedy++) vs brute (%s)", tag, cp2.error.c_str());
        if (cp2.ok && cp2.verts) { Dens rc = b.recount(cp2.set); CHECK(dens_cmp(rc, opt) == 0 && rc.num == cp2.tri, "%s: seeded witness recount", tag); }
        g_stats.seeded_flows += cp2.flow_calls; g_stats.unseeded_flows += ce.flow_calls;
        DsgResult cw2 = core_exact(g, core, false, &sg);                 // weak start + seed
        CHECK(cw2.ok && dens_cmp(cw2.dens(), opt) == 0, "%s: core_exact(weak start, seed) vs brute (%s)", tag, cw2.error.c_str());
        if (sg.verts) {
            DsgResult bad = sg; bad.tri += 1;                            // claims one triangle too many
            CHECK(!core_exact(g, core, true, &bad).ok, "%s: bogus seed (wrong triangle count) accepted", tag);
            DsgResult bad2 = sg; bad2.set.push_back(sg.set[0]); bad2.verts += 1;   // duplicate vertex
            CHECK(!core_exact(g, core, true, &bad2).ok, "%s: bogus seed (duplicate vertex) accepted", tag);
        }
    }

    // 4. Greedy / Greedy++
    DsgResult gr = greedy(g);
    CHECK(gr.ok, "%s: greedy: %s", tag, gr.error.c_str());
    CHECK(!dens_gt(gr.dens(), opt), "%s: greedy exceeds the optimum", tag);
    CHECK((u128)opt.num * gr.verts <= (u128)3 * gr.tri * opt.den || opt.num == 0, "%s: greedy below 1/3 of the optimum (%llu/%llu vs %llu/%llu)", tag, (unsigned long long)gr.tri, (unsigned long long)gr.verts, (unsigned long long)opt.num, (unsigned long long)opt.den);
    CHECK(!dens_gt(Dens{T_full, g.n}, gr.dens()), "%s: greedy worse than the whole graph", tag);
    { Dens rc = b.recount(gr.set); CHECK(gr.verts == 0 ? rc.den == 0 : rc.num == gr.tri && rc.den == gr.verts, "%s: greedy witness recount", tag); }
    {   // cross-check against the stage-5 greedy (independent code path, same tie-breaking)
        GreedyResult old = cpu_greedy_peel(g);
        CHECK(old.ok && dens_cmp(Dens{old.best_num, old.best_den}, gr.dens()) == 0, "%s: greedy differs from cpu_greedy_peel", tag);
    }
    DsgResult pp = greedy_pp(g, TMAX);
    CHECK(pp.ok, "%s: greedy_pp: %s", tag, pp.error.c_str());
    CHECK(pp.per_pass.size() == TMAX, "%s: per_pass size", tag);
    if (pp.per_pass.size() == TMAX) {
        CHECK(dens_cmp(pp.per_pass[0], gr.dens()) == 0, "%s: greedy_pp pass 1 != greedy", tag);
        for (uint32_t i = 0; i < TMAX; i++) {
            CHECK(!dens_gt(pp.per_pass[i], opt), "%s: greedy_pp pass %u exceeds the optimum", tag, i + 1);
            if (i) CHECK(dens_cmp(pp.per_pass[i], pp.per_pass[i - 1]) >= 0, "%s: greedy_pp not monotone at pass %u", tag, i + 1);
        }
        for (uint32_t t : {2u, 3u, 5u, 10u}) {            // standalone T-pass runs must reproduce the prefix of the long run
            DsgResult r = greedy_pp(g, t);
            CHECK(dens_cmp(r.dens(), pp.per_pass[t - 1]) == 0, "%s: greedy_pp(%u) != pass %u of the long run", tag, t, t);
        }
        { Dens rc = b.recount(pp.set); CHECK(pp.verts == 0 ? rc.den == 0 : rc.num == pp.tri && rc.den == pp.verts, "%s: greedy_pp witness recount", tag); }
        if (opt.num) {
            const bool gap = dens_cmp(gr.dens(), opt) < 0;
            g_stats.greedy_gap += gap;
            int first = -1;
            for (uint32_t i = 0; i < TMAX; i++) if (dens_cmp(pp.per_pass[i], opt) == 0) { first = (int)i + 1; break; }
            if (first > 0) { g_stats.pp_all_reached++; g_stats.first_T_hist[first == 1 ? 0 : first <= 2 ? 1 : first <= 5 ? 2 : first <= 10 ? 3 : 4]++; }
            if (gap && first > 0) g_stats.pp_reached++;
            if (dens_gt(pp.per_pass[TMAX - 1], gr.dens())) g_stats.pp_better_than_greedy++;
        }
    }

    // 5. pruning pipeline of gpu_dsg (with the CPU core numbers standing in for the GPU's, verified equal above)
    {
        Dens lb = core_level_density(g, core, kmax);
        if (dens_gt(gr.dens(), lb)) lb = gr.dens();
        CHECK(!dens_gt(lb, opt), "%s: lower bound exceeds optimum", tag);
        const uint64_t kp = dens_ceil(lb);
        Graph gp; std::vector<uint32_t> old_id, core_p;
        prune_to_core(g, core, kp, gp, old_id, core_p);
        std::vector<uint8_t> keep(g.n);
        for (uint32_t v = 0; v < g.n; v++) keep[v] = core[v] >= kp;
        EdgeList pe;
        std::vector<uint32_t> nid(g.n, UINT32_MAX);
        for (uint32_t i = 0; i < old_id.size(); i++) nid[old_id[i]] = i;
        for (auto& e : edges) if (keep[e.first] && keep[e.second]) pe.emplace_back(nid[e.first], nid[e.second]);
        validate_graph(gp, (uint32_t)old_id.size(), pe, "pruned");
        for (size_t i = 1; i < old_id.size(); i++) CHECK(old_id[i - 1] < old_id[i], "%s: old_id not ascending", tag);
        std::vector<uint32_t> chk;
        CHECK(core_decomposition(gp, chk) && chk == core_p, "%s: restricted cores != cores of the pruned graph", tag);
        DsgResult gpp_p = greedy_pp(gp, 10);
        DsgResult cp = core_exact(gp, core_p, true, &gpp_p);              // the pipeline of gpu_dsg: Greedy++ seeds CoreExact
        CHECK(cp.ok, "%s: core_exact(pruned, seeded): %s", tag, cp.error.c_str());
        CHECK(dens_cmp(cp.dens(), opt) == 0, "%s: pruned pipeline optimum %llu/%llu vs brute %llu/%llu", tag, (unsigned long long)cp.tri, (unsigned long long)cp.verts, (unsigned long long)opt.num, (unsigned long long)opt.den);
        // the witness (mapped back to original ids) must be a densest subgraph of the ORIGINAL graph
        std::vector<uint32_t> orig;
        for (uint32_t v : cp.set) orig.push_back(old_id[v]);
        if (cp.verts) { Dens rc = b.recount(orig); CHECK(dens_cmp(rc, opt) == 0, "%s: mapped witness is not optimal", tag); }
        g_stats.prune_nontrivial += gp.n < g.n;
    }

    // 6. induced_subgraph on random masks (also zero-size and full masks)
    for (int rep = 0; rep < 3; rep++) {
        std::vector<uint8_t> keep(g.n);
        const double p = rep == 0 ? 1.0 : (rep == 1 ? 0.0 : 0.6);
        for (auto& k : keep) k = urand() < p;
        Graph sub; std::vector<uint32_t> old_id;
        induced_subgraph(g, keep, sub, old_id);
        std::vector<uint32_t> nid(g.n, UINT32_MAX);
        for (uint32_t i = 0; i < old_id.size(); i++) nid[old_id[i]] = i;
        EdgeList se;
        uint32_t S = 0;
        for (auto& e : edges) if (keep[e.first] && keep[e.second]) se.emplace_back(nid[e.first], nid[e.second]);
        for (uint32_t v = 0; v < g.n; v++) if (keep[v]) S |= 1u << v;
        validate_graph(sub, (uint32_t)old_id.size(), se, "induced");
        CHECK(cpu_triangles_serial(sub) == b.tri_in(S), "%s: induced subgraph triangle count", tag);
        Graph rebuilt;
        build_graph_from_edges((uint32_t)old_id.size(), se, rebuilt);
        CHECK(graphs_equal(sub, rebuilt), "%s: induced_subgraph != build_graph_from_edges", tag);
    }
}

// ---- graph zoo ----
static EdgeList clique_edges(uint32_t base, uint32_t k) {
    EdgeList e;
    for (uint32_t u = 0; u < k; u++) for (uint32_t v = u + 1; v < k; v++) e.emplace_back(base + u, base + v);
    return e;
}
static void append(EdgeList& a, const EdgeList& b) { a.insert(a.end(), b.begin(), b.end()); }

static int run_tiny() {
    struct Spec { const char* name; uint32_t n; EdgeList e; };
    std::vector<Spec> zoo;
    zoo.push_back({"no edges", 5, {}});
    zoo.push_back({"path6", 6, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}}});
    { EdgeList e; for (uint32_t i = 1; i < 7; i++) e.emplace_back(0, i); zoo.push_back({"star7", 7, e}); }
    zoo.push_back({"C5", 5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}}});
    { EdgeList e; for (uint32_t a = 0; a < 3; a++) for (uint32_t c = 3; c < 6; c++) e.emplace_back(a, c); zoo.push_back({"K33 (no triangle)", 6, e}); }
    zoo.push_back({"single triangle + isolated", 7, {{0, 1}, {1, 2}, {0, 2}}});
    zoo.push_back({"triangle + pendant path", 6, {{0, 1}, {1, 2}, {0, 2}, {2, 3}, {3, 4}, {4, 5}}});
    for (uint32_t k : {3u, 4u, 5u, 6u, 7u, 9u, 12u}) zoo.push_back({"clique", k, clique_edges(0, k)});
    { EdgeList e = clique_edges(0, 4); append(e, clique_edges(4, 4)); zoo.push_back({"K4 + K4 (tie)", 8, e}); }
    { EdgeList e = clique_edges(0, 5); append(e, clique_edges(5, 4)); zoo.push_back({"K5 + K4 disjoint", 9, e}); }
    { EdgeList e = clique_edges(0, 5); append(e, clique_edges(5, 3)); append(e, clique_edges(8, 3)); zoo.push_back({"K5 + K3 + K3", 11, e}); }
    { EdgeList e = clique_edges(0, 5); append(e, clique_edges(4, 5)); zoo.push_back({"two K5 sharing a vertex", 9, e}); }
    { EdgeList e = clique_edges(0, 4); append(e, clique_edges(2, 4)); std::sort(e.begin(), e.end()); e.erase(std::unique(e.begin(), e.end()), e.end()); zoo.push_back({"two K4 sharing an edge", 6, e}); }
    { EdgeList e = clique_edges(0, 6); e.emplace_back(5, 6); e.emplace_back(6, 7); e.emplace_back(5, 7); zoo.push_back({"K6 + pendant triangle", 8, e}); }
    { EdgeList e; for (uint32_t i = 1; i <= 7; i++) { e.emplace_back(0, i); e.emplace_back(i, i % 7 + 1); } zoo.push_back({"wheel W8", 8, e}); }
    { EdgeList e; for (uint32_t i = 1; i < 8; i++) { e.emplace_back(0, i); if (i + 1 < 8) e.emplace_back(i, i + 1); } zoo.push_back({"fan", 8, e}); }
    { EdgeList e = clique_edges(0, 5); e.erase(std::find(e.begin(), e.end(), std::make_pair(0u, 1u))); zoo.push_back({"K5 minus edge", 5, e}); }
    zoo.push_back({"bowtie", 5, {{0, 1}, {1, 2}, {0, 2}, {2, 3}, {3, 4}, {2, 4}}});
    { EdgeList e = clique_edges(0, 4); append(e, clique_edges(4, 5)); e.emplace_back(3, 4); zoo.push_back({"K4 -- K5 bridge", 9, e}); }

    // graphs found by a random search where Charikar's greedy (ties by id) is strictly below the optimum, so that the
    // loads of Greedy++ are actually exercised (the optimum was computed by exact_densest at search time and is re-verified
    // below by brute force over all subsets)
    zoo.push_back({"greedy-gap graph", 10, {{0,6}, {1,2}, {1,4}, {1,7}, {1,8}, {2,8}, {3,4}, {3,6}, {4,6}, {6,7}, {7,8}, {7,9}}});
    zoo.push_back({"greedy-gap graph", 11, {{0,1}, {0,3}, {0,5}, {0,9}, {2,7}, {2,10}, {3,5}, {3,9}, {3,10}, {4,8}, {4,10}, {5,6}, {5,10}, {6,8}, {6,10}, {7,8}, {7,9}, {8,9}, {9,10}}});
    zoo.push_back({"greedy-gap graph", 12, {{0,4}, {0,8}, {0,11}, {1,3}, {1,4}, {1,5}, {1,10}, {2,3}, {2,5}, {2,6}, {2,8}, {2,10}, {2,11}, {3,4}, {3,5}, {3,7}, {3,10}, {4,8}, {4,9}, {4,10}, {5,6}, {5,7}, {5,8}, {6,8}, {6,10}, {7,10}, {8,9}, {8,10}, {8,11}, {9,11}}});
    zoo.push_back({"greedy-gap graph", 8, {{0,1}, {0,2}, {0,4}, {1,3}, {1,4}, {1,5}, {1,6}, {2,4}, {3,4}, {3,7}, {5,6}}});
    zoo.push_back({"greedy-gap graph", 11, {{0,1}, {0,2}, {0,4}, {0,5}, {0,6}, {0,9}, {1,2}, {1,8}, {2,4}, {2,7}, {2,9}, {2,10}, {3,7}, {3,8}, {4,8}, {5,6}, {7,9}, {8,10}}});
    zoo.push_back({"greedy-gap graph", 11, {{0,1}, {0,4}, {0,6}, {0,8}, {0,10}, {1,2}, {1,5}, {1,8}, {2,3}, {2,6}, {2,9}, {3,5}, {3,8}, {3,10}, {4,5}, {4,8}, {4,9}, {5,6}, {5,8}, {5,9}, {5,10}, {6,7}, {6,8}, {6,9}, {7,10}, {8,9}, {9,10}}});
    zoo.push_back({"greedy-gap graph", 9, {{0,2}, {0,4}, {1,6}, {1,8}, {2,3}, {2,4}, {2,5}, {2,7}, {2,8}, {3,4}, {3,7}, {6,8}}});
    zoo.push_back({"greedy-gap graph", 12, {{0,1}, {0,4}, {0,6}, {0,7}, {1,5}, {1,6}, {1,7}, {1,8}, {2,8}, {2,9}, {2,10}, {2,11}, {3,4}, {3,7}, {3,9}, {5,7}, {5,10}, {8,9}, {9,10}}});
    zoo.push_back({"greedy-gap graph", 11, {{0,3}, {0,4}, {0,8}, {0,9}, {0,10}, {1,2}, {1,4}, {1,5}, {1,8}, {2,3}, {2,4}, {2,6}, {3,5}, {3,6}, {4,7}, {5,8}, {5,9}, {5,10}, {6,7}, {6,8}, {6,10}, {7,9}, {7,10}, {9,10}}});

    int idx = 0;
    auto run = [&](uint32_t n, EdgeList e) {
        for (auto& x : e) if (x.first > x.second) std::swap(x.first, x.second);
        std::sort(e.begin(), e.end());
        e.erase(std::unique(e.begin(), e.end()), e.end());
        Graph g;
        build_graph_from_edges(n, e, g);
        Brute b(g);
        test_graph(g, b, e, idx, idx % 4 == 0);
        if (idx % 10 == 0) {                                // load_graph path: same Graph from a file
            FILE* f = fopen("bin/_tiny_dsg.txt", "w");
            for (auto& x : e) fprintf(f, "%u %u\n", x.first, x.second);
            if (e.empty()) fprintf(f, "# empty\n");
            fclose(f);
            Graph g2;
            if (!e.empty() && load_graph("bin/_tiny_dsg.txt", g2) && g2.n == n)   // load_graph infers n = max id + 1
                CHECK(graphs_equal(g, g2), "load_graph != build_graph_from_edges (graph #%d)", idx);
        }
        idx++;
    };
    for (auto& z : zoo) { run(z.n, z.e); }
    const double ps[] = {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9};
    for (int iter = 0; iter < 450; iter++) {
        const uint32_t n = 3 + rng() % 10;                  // 3..12
        const double p = ps[iter % 9];
        EdgeList e;
        const int kind = iter % 5;
        if (kind == 4 && n >= 6) {                           // disjoint union of two random graphs (different densities)
            const uint32_t h = n / 2;
            for (uint32_t u = 0; u < n; u++) for (uint32_t v = u + 1; v < n; v++)
                if ((u < h) == (v < h) && urand() < (u < h ? p : std::min(0.95, p + 0.3))) e.emplace_back(u, v);
        } else {
            for (uint32_t u = 0; u < n; u++) for (uint32_t v = u + 1; v < n; v++) if (urand() < p) e.emplace_back(u, v);
            if (kind == 0) { uint32_t k = 3 + rng() % 4; append(e, clique_edges(rng() % (n > k ? n - k + 1 : 1), std::min(k, n))); }   // planted clique
        }
        run(n, e);
        if (g_fail > 25) break;
    }
    printf("tiny mode: %d graphs (%d with triangles, %d without); all 2^n subsets brute-forced for each\n", g_stats.graphs, g_stats.with_tri, g_stats.no_tri);
    printf("  Greedy: strictly below optimum on %d graphs; Greedy++ (T=%u) reached the optimum on %d of the %d graphs with triangles,\n"
           "          and on %d of those %d greedy-gap graphs; better than greedy on %d graphs\n",
           g_stats.greedy_gap, TMAX, g_stats.pp_all_reached, g_stats.with_tri, g_stats.pp_reached, g_stats.greedy_gap, g_stats.pp_better_than_greedy);
    printf("  first pass that hit the optimum: pass1 %d, pass2 %d, pass3-5 %d, pass6-10 %d, pass11-30 %d\n", g_stats.first_T_hist[0], g_stats.first_T_hist[1], g_stats.first_T_hist[2], g_stats.first_T_hist[3], g_stats.first_T_hist[4]);
    printf("  pruned graph strictly smaller than the input on %d graphs\n", g_stats.prune_nontrivial);
    printf("  max-flow calls over the tiny suite: unseeded %lld, seeded with Greedy++ %lld\n", g_stats.unseeded_flows, g_stats.seeded_flows);
    printf("  CoreExact with weak start: line-17 shrink used on %d graphs, >1 component searched on %d graphs\n", g_stats.weak_shrinks, g_stats.weak_multi);
    CHECK(g_stats.with_tri >= 250 && g_stats.no_tri >= 15, "too few graphs with / without triangles to be a meaningful test");
    CHECK(g_stats.greedy_gap >= 3 && g_stats.pp_better_than_greedy >= 1, "the suite contains no graph where Greedy++ improves on greedy (the loads would be untested)");
    CHECK(g_stats.weak_shrinks >= 10 && g_stats.weak_multi >= 10, "the weak-start CoreExact does not exercise the shrink / multi-component paths on enough graphs");
    CHECK(g_stats.prune_nontrivial >= 20, "pruning rarely removes anything in this suite");
    return 0;
}

// ---- medium random graphs with several planted communities of different density (n up to ~100): no brute force possible,
// so Exact (Alg 1, whole graph) is the oracle for CoreExact (Alg 4, components + shrinking) and for the pruned pipeline ----
static int run_mid() {
    int graphs = 0, flows_gt1 = 0, pruned_smaller = 0, pp_hit = 0;
    for (int iter = 0; iter < 160; iter++) {
        const uint32_t n = (iter % 4 == 1 ? 50 : 20) + rng() % 80;
        EdgeList e;
        uint32_t base = 0;
        const int ncomm = 1 + rng() % 4;
        for (int c = 0; c < ncomm && base + 4 < n; c++) {
            const bool er_block = iter % 4 == 1;           // every 4th graph: disjoint ER blocks (inhomogeneous, no single clique)
            const uint32_t sz = std::min<uint32_t>(er_block ? 12 + rng() % 14 : 4 + rng() % 11, n - base);
            const double pc = er_block ? 0.25 + 0.3 * urand() : 0.45 + 0.55 * urand();
            for (uint32_t u = 0; u < sz; u++) for (uint32_t v = u + 1; v < sz; v++) if (urand() < pc) e.emplace_back(base + u, base + v);
            base += sz;
        }
        const double noise = iter % 3 == 0 || iter % 4 == 1 ? 0.0 : 0.01 + 0.06 * urand();      // every third graph: communities stay disconnected
        for (uint32_t u = 0; u < n; u++) for (uint32_t v = u + 1; v < n; v++) if (urand() < noise) e.emplace_back(u, v);
        std::sort(e.begin(), e.end());
        e.erase(std::unique(e.begin(), e.end()), e.end());
        Graph g;
        build_graph_from_edges(n, e, g);
        const uint64_t T_full = cpu_triangles_serial(g);
        graphs++;
        std::vector<uint32_t> core; uint32_t kmax = 0;
        CHECK(core_decomposition(g, core, &kmax), "core_decomposition failed");
        CHECK(core == core_by_definition(g), "mid #%d: Alg 3 differs from the fixpoint definition", iter);
        if (iter % 4 == 0) {
            EmuCoreBackend be(g, 3, 64);
            CoreLoopResult r = core_decomp_loop(g.n, T_full, be);
            CHECK(r.ok && be.core == core, "mid #%d: emulated GPU cores differ", iter);
        }
        DsgResult ex = exact_densest(g), ce = core_exact(g, core), gr = greedy(g), pp = greedy_pp(g, 15);
        CHECK(ex.ok && ce.ok && gr.ok && pp.ok, "mid #%d: algorithm failed: %s %s", iter, ex.error.c_str(), ce.error.c_str());
        { DsgResult cw = core_exact(g, core, false); CHECK(cw.ok && dens_cmp(cw.dens(), ex.dens()) == 0, "mid #%d: core_exact(weak start) %llu/%llu vs exact", iter, (unsigned long long)cw.tri, (unsigned long long)cw.verts); }
        { DsgResult cs = core_exact(g, core, true, &pp); CHECK(cs.ok && dens_cmp(cs.dens(), ex.dens()) == 0, "mid #%d: core_exact(seed=greedy++) vs exact (%s)", iter, cs.error.c_str());
          g_stats.seeded_flows += cs.flow_calls; g_stats.unseeded_flows += ce.flow_calls; }
        CHECK(dens_cmp(ex.dens(), ce.dens()) == 0, "mid #%d: exact %llu/%llu vs core_exact %llu/%llu", iter, (unsigned long long)ex.tri, (unsigned long long)ex.verts, (unsigned long long)ce.tri, (unsigned long long)ce.verts);
        for (const DsgResult* r : {&ex, &ce, &gr, &pp}) {
            std::vector<uint8_t> mask(g.n, 0);
            for (uint32_t v : r->set) mask[v] = 1;
            CHECK(count_triangles_in(g, mask) == r->tri && r->set.size() == r->verts, "mid #%d: witness recount failed", iter);
        }
        CHECK(!dens_gt(gr.dens(), ex.dens()) && !dens_gt(pp.dens(), ex.dens()), "mid #%d: heuristic above optimum", iter);
        CHECK((u128)ex.tri * gr.verts <= (u128)3 * gr.tri * ex.verts, "mid #%d: greedy below 1/3 optimum", iter);
        for (size_t i = 1; i < pp.per_pass.size(); i++) CHECK(dens_cmp(pp.per_pass[i], pp.per_pass[i - 1]) >= 0, "mid #%d: greedy_pp not monotone", iter);
        Dens lb = core_level_density(g, core, kmax);
        if (dens_gt(gr.dens(), lb)) lb = gr.dens();
        Graph gp; std::vector<uint32_t> old_id, core_p;
        prune_to_core(g, core, dens_ceil(lb), gp, old_id, core_p);
        std::vector<uint32_t> chk;
        CHECK(core_decomposition(gp, chk) && chk == core_p, "mid #%d: restricted cores != cores of pruned graph", iter);
        DsgResult cp = core_exact(gp, core_p);
        CHECK(cp.ok && dens_cmp(cp.dens(), ex.dens()) == 0, "mid #%d: pruned pipeline optimum differs", iter);
        pruned_smaller += gp.n < g.n;
        flows_gt1 += ce.flow_calls > 1;
        pp_hit += dens_cmp(pp.dens(), ex.dens()) == 0;
        if (g_fail > 25) break;
    }
    // Phase 2: disjoint ER blocks of different density. Searching MORE THAN ONE component (lines 5-20 with a second pass through
    // lines 6-9 and 10-19) and the line-17 shrink are rare in random graphs, so generate many such graphs cheaply (CoreExact
    // only) and verify the interesting ones against Exact (Alg 1).
    int targeted = 0, t_multi = 0, t_shrink = 0, weak_checked = 0, w_shrunk = 0, w_multi = 0;
    for (int iter = 0; iter < 400; iter++) {
        EdgeList e;
        uint32_t base = 0;
        const int nb = 2 + rng() % 3;
        for (int b = 0; b < nb; b++) {
            const uint32_t sz = 14 + rng() % 20;
            const double p = 0.12 + 0.3 * urand();
            for (uint32_t u = 0; u < sz; u++) for (uint32_t v = u + 1; v < sz; v++) if (urand() < p) e.emplace_back(base + u, base + v);
            base += sz;
        }
        Graph g;
        build_graph_from_edges(base, e, g);
        std::vector<uint32_t> core;
        core_decomposition(g, core);
        DsgResult ce = core_exact(g, core);
        CHECK(ce.ok, "blocks #%d: core_exact failed: %s", iter, ce.error.c_str());
        {
            DsgResult cw = core_exact(g, core, false);
            CHECK(cw.ok && dens_cmp(cw.dens(), ce.dens()) == 0, "blocks #%d: core_exact weak start %llu/%llu vs normal %llu/%llu", iter, (unsigned long long)cw.tri, (unsigned long long)cw.verts, (unsigned long long)ce.tri, (unsigned long long)ce.verts);
            if (cw.shrinks > 0 || cw.components_searched > 1) {      // interesting weak-start run: confirm against Exact too
                DsgResult ex = exact_densest(g);
                CHECK(ex.ok && dens_cmp(ex.dens(), cw.dens()) == 0, "blocks #%d: exact %llu/%llu vs weak-start core_exact", iter, (unsigned long long)ex.tri, (unsigned long long)ex.verts);
                weak_checked++; w_shrunk += cw.shrinks > 0; w_multi += cw.components_searched > 1;
            }
        }
        if (!(ce.components_searched > 1 || ce.shrinks > 0) || targeted >= 40) continue;
        targeted++; t_multi += ce.components_searched > 1; t_shrink += ce.shrinks > 0;
        DsgResult ex = exact_densest(g);
        CHECK(ex.ok && dens_cmp(ex.dens(), ce.dens()) == 0, "blocks #%d: exact %llu/%llu vs core_exact %llu/%llu", iter, (unsigned long long)ex.tri, (unsigned long long)ex.verts, (unsigned long long)ce.tri, (unsigned long long)ce.verts);
        std::vector<uint8_t> mask(g.n, 0);
        for (uint32_t v : ce.set) mask[v] = 1;
        CHECK(count_triangles_in(g, mask) == ce.tri && ce.set.size() == ce.verts, "blocks #%d: witness recount failed", iter);
        if (g_fail > 25) break;
    }
    printf("  blocks phase: %d graphs checked against Exact with the normal start (%d searched >1 component, %d used the line-17 shrink); %d more with the weak start (%d used the shrink, %d searched >1 component)\n", targeted, t_multi, t_shrink, weak_checked, w_shrunk, w_multi);
    CHECK(weak_checked >= 40, "weak-start runs that shrink / search several components are too rare in the blocks phase");
    printf("mid mode: %d graphs (planted communities); CoreExact needed >1 flow call on %d; pruning shrank %d; Greedy++(T=15) hit the optimum on %d\n", graphs, flows_gt1, pruned_smaller, pp_hit);
    CHECK(flows_gt1 >= 20, "too few graphs exercise the binary search of CoreExact");
    CHECK(w_shrunk >= 20 && w_multi >= 20, "the weak-start runs do not exercise the line-17 shrink / multi-component paths of CoreExact enough");
    return 0;
}

static int run_file(const char* path) {
    Graph g;
    if (!load_graph(path, g)) return 1;
    const uint64_t T_full = cpu_triangles_serial(g);
    printf("file mode: n=%u m=%u triangles=%llu\n", g.n, g.m, (unsigned long long)T_full);
    EdgeList edges;
    for (uint32_t e = 0; e < g.m; e++) edges.emplace_back(std::min(g.src[e], g.col[e]), std::max(g.src[e], g.col[e]));
    validate_graph(g, g.n, edges, "loaded");

    std::vector<uint32_t> core; uint32_t kmax = 0;
    CHECK(core_decomposition(g, core, &kmax), "core_decomposition failed");
    CHECK(core == core_by_definition(g), "Alg 3 differs from the fixpoint definition");
    struct Cfg { unsigned grid, block; } cfgs[] = {{3, 64}, {1, 32}};
    for (auto c : cfgs) {
        EmuCoreBackend be(g, c.grid, c.block);
        CoreLoopResult r = core_decomp_loop(g.n, T_full, be);
        CHECK(r.ok, "emulated core loop: %s", r.error.c_str());
        CHECK(be.core == core, "emulated GPU core numbers differ from CPU Alg 3 (grid %u block %u)", c.grid, c.block);
        printf("  emulated GPU core loop (grid %u block %u): %u rounds, kmax %u\n", c.grid, c.block, r.rounds, r.kmax);
    }
    DsgResult ex = exact_densest(g), ce = core_exact(g, core), gr = greedy(g), pp = greedy_pp(g, 20);
    CHECK(ex.ok && ce.ok && gr.ok && pp.ok, "an algorithm failed: %s %s %s %s", ex.error.c_str(), ce.error.c_str(), gr.error.c_str(), pp.error.c_str());
    CHECK(dens_cmp(ex.dens(), ce.dens()) == 0, "exact %llu/%llu vs core_exact %llu/%llu", (unsigned long long)ex.tri, (unsigned long long)ex.verts, (unsigned long long)ce.tri, (unsigned long long)ce.verts);
    for (const DsgResult* r : {&ex, &ce, &gr, &pp}) {        // witnesses re-counted with the stage-5 triangle counter
        std::vector<uint8_t> mask(g.n, 0);
        for (uint32_t v : r->set) mask[v] = 1;
        CHECK(count_triangles_in(g, mask) == r->tri && r->set.size() == r->verts, "witness recount failed");
        std::vector<uint32_t> tri;
        CHECK(cpu_triangle_participation(g, mask, tri) == r->tri, "witness recount (stage-5 counter) failed");
    }
    CHECK(!dens_gt(gr.dens(), ce.dens()) && !dens_gt(pp.dens(), ce.dens()), "heuristic above the optimum");
    CHECK((u128)ce.tri * gr.verts <= (u128)3 * gr.tri * ce.verts, "greedy below 1/3 of the optimum");
    for (size_t i = 1; i < pp.per_pass.size(); i++) CHECK(dens_cmp(pp.per_pass[i], pp.per_pass[i - 1]) >= 0, "greedy_pp not monotone");
    {
        GreedyResult old = cpu_greedy_peel(g);
        CHECK(old.ok && dens_cmp(Dens{old.best_num, old.best_den}, gr.dens()) == 0, "greedy differs from cpu_greedy_peel");
    }
    // pruning pipeline
    Dens lb = core_level_density(g, core, kmax);
    if (dens_gt(gr.dens(), lb)) lb = gr.dens();
    Graph gp; std::vector<uint32_t> old_id, core_p;
    prune_to_core(g, core, dens_ceil(lb), gp, old_id, core_p);
    std::vector<uint32_t> chk;
    CHECK(core_decomposition(gp, chk) && chk == core_p, "restricted cores != cores of the pruned graph");
    DsgResult cp = core_exact(gp, core_p);
    CHECK(cp.ok && dens_cmp(cp.dens(), ce.dens()) == 0, "pruned pipeline optimum differs from the full-graph optimum");
    printf("  optimum %llu/%llu = %.6f (exact == core_exact == pruned pipeline); greedy %.6f, greedy++(T=20) %.6f; lb %.6f -> pruned n=%u of %u\n",
           (unsigned long long)ce.tri, (unsigned long long)ce.verts, dens_value(ce.dens()), dens_value(gr.dens()), dens_value(pp.dens()), dens_value(lb), gp.n, g.n);
    return 0;
}


// ---- hub graphs: the incremental update kernel gives each frontier vertex to ONE warp whose 32 lanes split its neighbour list.
// Random small graphs never put a vertex with > 32 live neighbours into the frontier, so lane striding / list positions >= 32
// would go untested. Here a hub has 40-100 neighbours, almost all of them leaves, and 2-4 of them sit in a dense cluster, so
// the hub has few triangles (it is peeled early, in the first frontier) while its triangle partners stay alive with
// tri[] that must be decremented. Vertex ids are shuffled so the cluster members land at arbitrary positions of the hub's
// sorted neighbour list (including positions >= 32). Oracle: CPU Alg 3 (itself brute-force verified on the tiny suite). ----
static int run_hub() {
    int graphs = 0, hub_in_first_frontier = 0, wide_hubs = 0;
    for (int iter = 0; iter < 120; iter++) {
        const uint32_t leaves = 40 + rng() % 61, cl = 8 + rng() % 8, hubs = 1 + rng() % 3;
        const uint32_t n = hubs + leaves + cl;
        std::vector<uint32_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0u);
        std::shuffle(perm.begin(), perm.end(), rng);
        EdgeList e;
        auto E = [&](uint32_t a, uint32_t b) { e.emplace_back(perm[a], perm[b]); };
        const uint32_t cl0 = hubs + leaves;                       // dense cluster = ids [cl0, n)
        for (uint32_t a = cl0; a < n; a++) for (uint32_t b = a + 1; b < n; b++) if (urand() < 0.7) E(a, b);
        for (uint32_t h = 0; h < hubs; h++) {
            for (uint32_t l = 0; l < leaves; l++) if (urand() < 0.9) E(hubs + l, h);       // leaves: degree 1 (plus maybe the other hubs)
            const uint32_t t = 2 + rng() % 3;
            for (uint32_t i = 0; i < t; i++) E(h, cl0 + rng() % cl);
        }
        Graph g;
        build_graph_from_edges(n, e, g);
        std::vector<uint32_t> core, tri;
        uint32_t kmax = 0;
        CHECK(core_decomposition(g, core, &kmax), "hub #%d: core_decomposition failed", iter);
        const uint64_t T_full = cpu_triangles_serial(g);
        std::vector<uint32_t> urow, ucol;
        build_undirected_csr(g, urow, ucol);
        uint32_t maxdeg = 0;
        for (uint32_t v = 0; v < n; v++) maxdeg = std::max(maxdeg, urow[v + 1] - urow[v]);
        wide_hubs += maxdeg >= 33;
        struct Cfg { unsigned grid, block; } cfgs[] = {{1, 32}, {2, 64}};
        for (auto& cf : cfgs) {
            EmuCoreBackendInc be(g, cf.grid, cf.block);
            CoreLoopResult r = core_decomp_loop_inc(g.n, T_full, be);
            CHECK(r.ok, "hub #%d: incremental loop failed: %s", iter, r.error.c_str());
            CHECK(be.core == core, "hub #%d: incremental core numbers differ from CPU Alg 3 (grid %u block %u)", iter, cf.grid, cf.block);
        }
        graphs++;
        if (g_fail > 25) break;
    }
    printf("hub mode: %d graphs, %d with a vertex of degree >= 33\n", graphs, wide_hubs);
    CHECK(wide_hubs >= 100, "too few graphs with a wide hub");
    (void)hub_in_first_frontier;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    int rc = strcmp(argv[1], "--tiny") == 0 ? run_tiny() : strcmp(argv[1], "--mid") == 0 ? run_mid() : strcmp(argv[1], "--hub") == 0 ? run_hub() : run_file(argv[1]);
    if (rc) return rc;
    printf(g_fail ? "FAILED (%d checks)\n" : "ALL OK\n", g_fail);
    return g_fail ? 1 : 0;
}
