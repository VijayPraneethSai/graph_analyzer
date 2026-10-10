// densest.h - triangle-density densest subgraph: CPU reference algorithms + the host-side peeling loop.
//
// PROBLEM. Find a vertex set S maximising  density(S) = triangles(S) / |S|,  where triangles(S) counts the
// triangles whose three corners are all in S (the "k=3 clique densest subgraph" problem).
//
// BATCH PEELING (Bahmani-Kumar-Vassilvitskii for edges, generalised to triangles). Repeat:
//   1. count T = triangles in the subgraph induced by the alive vertices and tri[v] = number of those
//      triangles containing v;
//   2. density T/nalive: remember it if it beats the best so far (fractions compared by cross multiplication
//      in uint64, no floating point);
//   3. remove EVERY alive vertex with tri[v] <= (1+eps) * 3T/nalive   (3T/nalive = average participation,
//      because sum_v tri[v] = 3T);
//   until nothing is alive. removed_round[v] = round in which v was removed, so the best subgraph is
//   { v : removed_round[v] >= best_round }  (the alive set at the START of round best_round).
//
// WHY IT TERMINATES FAST. The minimum of tri[] is <= the average <= (1+eps)*average, so every round removes
// at least one vertex. By Markov, fewer than 1/(1+eps) of the vertices can have tri[v] > (1+eps)*average, so
// each round removes at least a fraction eps/(1+eps) of the alive vertices: at most about log_{1+eps}(n)
// = O(log n / eps) rounds.
//
// APPROXIMATION GUARANTEE (taken from the literature, sketched here, not proven in this repo): the best
// density seen is within a factor 3(1+eps) of the optimum. Sketch: let S* be an optimal set and rho* its
// density. Removing any v from S* cannot raise the density, which gives  tri_{S*}(v) >= rho*  for all v in S*.
// Look at the first round in which a vertex v of S* is removed; the alive set S then contains S*, so
// tri_S(v) >= tri_{S*}(v) >= rho*, while v was removed because tri_S(v) <= 3(1+eps) * density(S). Hence
// density(S) >= rho* / (3(1+eps)). The k-clique generalisation of the Bahmani et al. peeling is due to
// Tsourakakis (2015); I quote that from memory - check the paper before citing it. The guarantee is a WORST
// case bound: it says nothing about how close to optimal the result is on a particular graph, and the
// emulator test only confirms it on tiny random graphs (n <= 16) by brute force.
//
// CPU GREEDY (cpu_greedy_peel) removes ONE vertex at a time, always the one with the fewest triangles.
// It is the classical sequential peeling and a (1/3)-approximation for triangle density; it is the quality
// baseline for the batch version. It is inherently sequential, which is what the batch version fixes.
//
// ARITHMETIC (all uint64, no floating point in any decision). With T0 = triangles of the whole graph, n vertices,
// eps = num/den:
//   tri[v] <= T <= T0   and must fit uint32 (GPU counters)     -> T0 < 2^32
//   removal test:  tri[v] * nalive * den  <=  3 * (den+num) * T -> both sides must be < 2^64, checked by
//                                                                  peel_arith_ok() (e.g. 1e8 * 4e6 * 10 = 4e15)
//   density compare: T * best_den  vs  best_num * nalive         -> each factor < 2^32, product < 2^64
#pragma once
#include <queue>
#include <string>
#include "graph.h"

struct PeelResult {
    uint64_t best_num = 0;          // triangles of the best subgraph
    uint64_t best_den = 0;          // vertices of the best subgraph (0 = nothing recorded yet)
    uint32_t best_round = 0;        // best subgraph = { v : removed_round[v] >= best_round }
    uint32_t rounds = 0;            // rounds executed (until no alive vertex is left)
    std::vector<uint32_t> removed_round;
    bool ok = true;                 // false: the run was aborted, see error
    std::string error;
};

inline bool operator==(const PeelResult& a, const PeelResult& b) {
    return a.best_num == b.best_num && a.best_den == b.best_den && a.best_round == b.best_round &&
           a.rounds == b.rounds && a.removed_round == b.removed_round;
}

// Display value only. All decisions use integers.
inline double density_value(uint64_t num, uint64_t den) { return den ? (double)num / (double)den : 0.0; }

// Checks every bound listed at the top of this file for a graph with n vertices and T0 triangles.
inline bool peel_arith_ok(uint32_t n, uint64_t T0, uint64_t num, uint64_t den, std::string* why = nullptr) {
    auto fail = [&](const char* msg) { if (why) *why = msg; return false; };
    if (den < 1 || den > (1ull << 20) || num > (1ull << 20)) return fail("eps = num/den needs 1 <= den <= 2^20 and num <= 2^20");
    if (T0 >= (1ull << 32)) return fail("triangle count >= 2^32: uint32 tri[] counters could overflow");
    if (n > 0 && T0 > UINT64_MAX / ((uint64_t)n * den)) return fail("tri*nalive*den could overflow uint64");
    if (T0 > UINT64_MAX / (3ull * (den + num))) return fail("3*(den+num)*T could overflow uint64");
    return true;
}

// true if density T/nalive is strictly better than best_num/best_den (best_den == 0: nothing recorded yet).
// Ties keep the EARLIER (larger) subgraph. Needs T < 2^32 and nalive < 2^32 (see peel_arith_ok).
inline bool peel_improves(uint64_t T, uint64_t nalive, uint64_t best_num, uint64_t best_den) {
    return best_den == 0 || T * best_den > best_num * nalive;
}

// ---- triangle participation in the subgraph induced by alive[] (serial; same semantics as the GPU kernel) ----
inline uint64_t cpu_triangle_participation(const Graph& g, const std::vector<uint8_t>& alive,
                                           std::vector<uint32_t>& tri) {
    tri.assign(g.n, 0);
    uint64_t total = 0;
    for (uint32_t e = 0; e < g.m; e++) {
        const uint32_t u = g.src[e], v = g.col[e];
        if (!alive[u] || !alive[v]) continue;
        uint32_t i = g.rowptr[u], iend = g.rowptr[u + 1], j = g.rowptr[v], jend = g.rowptr[v + 1];
        while (i < iend && j < jend) {
            const uint32_t a = g.col[i], b = g.col[j];
            if (a < b) i++;
            else if (a > b) j++;
            else {
                const uint32_t w = a;
                if (alive[w]) { total++; tri[u]++; tri[v]++; tri[w]++; }
                i++; j++;
            }
        }
    }
    return total;
}

// OpenMP version (edge-parallel, atomic increments). Identical results; without -fopenmp it is serial.
inline uint64_t cpu_triangle_participation_omp(const Graph& g, const std::vector<uint8_t>& alive,
                                               std::vector<uint32_t>& tri) {
    tri.assign(g.n, 0);
    uint64_t total = 0;
#pragma omp parallel for schedule(dynamic, 1024) reduction(+ : total)
    for (long long ei = 0; ei < (long long)g.m; ei++) {
        const uint32_t e = (uint32_t)ei;
        const uint32_t u = g.src[e], v = g.col[e];
        if (!alive[u] || !alive[v]) continue;
        uint32_t i = g.rowptr[u], iend = g.rowptr[u + 1], j = g.rowptr[v], jend = g.rowptr[v + 1];
        while (i < iend && j < jend) {
            const uint32_t a = g.col[i], b = g.col[j];
            if (a < b) i++;
            else if (a > b) j++;
            else {
                const uint32_t w = a;
                if (alive[w]) {
                    total++;
#pragma omp atomic
                    tri[u]++;
#pragma omp atomic
                    tri[v]++;
#pragma omp atomic
                    tri[w]++;
                }
                i++; j++;
            }
        }
    }
    return total;
}

// ---- CPU batch peeling: the bit-exact reference for the GPU pipeline --------------------------------------
// Same rounds, same integer test, same tie-breaking as the GPU loop (batch_peel_loop below).
inline PeelResult cpu_batch_peel(const Graph& g, uint64_t num, uint64_t den, bool use_omp = false) {
    PeelResult r;
    r.removed_round.assign(g.n, UINT32_MAX);
    std::vector<uint8_t> alive(g.n, 1);
    std::vector<uint32_t> tri;
    uint64_t nalive = g.n;
    for (uint32_t round = 0; nalive > 0; round++) {
        const uint64_t T = use_omp ? cpu_triangle_participation_omp(g, alive, tri) : cpu_triangle_participation(g, alive, tri);
        if (round == 0) {
            std::string why;
            if (!peel_arith_ok(g.n, T, num, den, &why)) { r.ok = false; r.error = why; return r; }
        }
        if (peel_improves(T, nalive, r.best_num, r.best_den)) { r.best_num = T; r.best_den = nalive; r.best_round = round; }
        const uint64_t rhs = 3ull * (den + num) * T;
        uint64_t removed = 0;
        for (uint32_t v = 0; v < g.n; v++)
            if (alive[v] && (uint64_t)tri[v] * nalive * den <= rhs) { alive[v] = 0; r.removed_round[v] = round; removed++; }
        if (removed == 0) { r.ok = false; r.error = "no vertex removed in a round (impossible if the maths holds)"; return r; }
        nalive -= removed;
        r.rounds = round + 1;
    }
    return r;
}

// ---- undirected CSR (both directions, sorted lists) built from the oriented graph -----------------------
inline void build_undirected_csr(const Graph& g, std::vector<uint32_t>& urow, std::vector<uint32_t>& ucol) {
    urow.assign(g.n + 1, 0);
    for (uint32_t e = 0; e < g.m; e++) { urow[g.src[e] + 1]++; urow[g.col[e] + 1]++; }
    for (uint32_t i = 0; i < g.n; i++) urow[i + 1] += urow[i];
    ucol.resize(2ull * g.m);
    std::vector<uint32_t> fill(urow.begin(), urow.end() - 1);
    for (uint32_t e = 0; e < g.m; e++) { ucol[fill[g.src[e]]++] = g.col[e]; ucol[fill[g.col[e]]++] = g.src[e]; }
    for (uint32_t u = 0; u < g.n; u++) std::sort(ucol.begin() + urow[u], ucol.begin() + urow[u + 1]);
}

// Calls f(w) for every w > lo present in both sorted ranges A[0..na) and B[0..nb). Merge, or binary search of
// the shorter range in the longer one when the lengths differ by more than 16x (hub vertices).
template <class F>
inline void for_common_above(const uint32_t* A, uint32_t na, const uint32_t* B, uint32_t nb, uint32_t lo, F&& f) {
    const uint32_t *ap = std::upper_bound(A, A + na, lo), *ae = A + na;
    const uint32_t *bp = std::upper_bound(B, B + nb, lo), *be = B + nb;
    const size_t sa = (size_t)(ae - ap), sb = (size_t)(be - bp);
    if (sa * 16 < sb) {
        for (; ap < ae; ++ap) { bp = std::lower_bound(bp, be, *ap); if (bp == be) break; if (*bp == *ap) f(*ap); }
    } else if (sb * 16 < sa) {
        for (; bp < be; ++bp) { ap = std::lower_bound(ap, ae, *bp); if (ap == ae) break; if (*ap == *bp) f(*bp); }
    } else {
        while (ap < ae && bp < be) {
            if (*ap < *bp) ++ap;
            else if (*ap > *bp) ++bp;
            else { f(*ap); ++ap; ++bp; }
        }
    }
}

struct GreedyResult {
    uint64_t best_num = 0, best_den = 0;   // best density seen = best_num / best_den
    bool ok = true;
    std::string error;
};

// ---- CPU greedy peeling (quality baseline) -------------------------------------------------------------
// Repeat: record density T/nalive if strictly better; remove the alive vertex with the fewest triangles
// (ties: smaller id, via the heap order); every triangle {v,a,b} through the removed v lowers cur[a], cur[b] by 1.
// Priority queue with LAZY deletion: entries are (cur value, vertex); an entry is stale if the vertex is dead or its
// cur value has changed since the push. cur[] only decreases, so the latest entry of an alive vertex is always
// valid. We push each touched vertex once per removal (not once per triangle): at most 2m pushes in total.
// Triangles through v are enumerated by intersecting the sorted UNDIRECTED lists N(v) and N(a) for every alive
// neighbour a (only b > a, so each triangle once) and skipping dead b. Cost per removal is about
// sum over alive neighbours a of (deg v + deg a) - fine for the small/medium graphs, too slow for graphs with huge hubs
// (gpu_densest has --no-greedy). Self-check: the enumerated triangle count must equal cur[v].
inline GreedyResult cpu_greedy_peel(const Graph& g) {
    GreedyResult r;
    std::vector<uint32_t> urow, ucol;
    build_undirected_csr(g, urow, ucol);
    std::vector<uint8_t> alive(g.n, 1);
    std::vector<uint32_t> tri32;
    uint64_t T = cpu_triangle_participation(g, alive, tri32);
    if (T >= (1ull << 32)) { r.ok = false; r.error = "greedy: triangle count >= 2^32, density compare could overflow"; return r; }
    std::vector<uint64_t> cur(tri32.begin(), tri32.end());
    typedef std::pair<uint64_t, uint32_t> Entry;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    for (uint32_t v = 0; v < g.n; v++) heap.emplace(cur[v], v);
    uint64_t nalive = g.n;
    std::vector<uint32_t> touched;
    std::vector<uint8_t> is_touched(g.n, 0);
    while (nalive > 0) {
        Entry top = heap.top(); heap.pop();
        const uint32_t v = top.second;
        if (!alive[v] || cur[v] != top.first) continue;          // stale entry
        if (peel_improves(T, nalive, r.best_num, r.best_den)) { r.best_num = T; r.best_den = nalive; }
        uint64_t found = 0;
        touched.clear();
        const uint32_t* nv = ucol.data() + urow[v];
        const uint32_t dv = urow[v + 1] - urow[v];
        for (uint32_t k = 0; k < dv; k++) {
            const uint32_t a = nv[k];
            if (!alive[a]) continue;
            for_common_above(nv, dv, ucol.data() + urow[a], urow[a + 1] - urow[a], a, [&](uint32_t b) {
                if (!alive[b]) return;
                found++;
                if (cur[a] == 0 || cur[b] == 0) { r.ok = false; return; }
                cur[a]--; cur[b]--;
                if (!is_touched[a]) { is_touched[a] = 1; touched.push_back(a); }
                if (!is_touched[b]) { is_touched[b] = 1; touched.push_back(b); }
            });
        }
        if (!r.ok || found != cur[v]) { r.ok = false; r.error = "greedy: triangle bookkeeping inconsistent"; return r; }
        alive[v] = 0;
        T -= cur[v];
        nalive--;
        for (uint32_t x : touched) { is_touched[x] = 0; heap.emplace(cur[x], x); }
    }
    return r;
}

// One CSV row: impl,graph,n,m,best_triangles,best_vertices,density,rounds,time_ms,h2d_ms,total_ms
// (density is a display value with 6 decimals; every comparison in this project uses the integers).
inline std::string densest_row(const char* impl, const char* graph, uint32_t n, uint32_t m, uint64_t best_num,
                               uint64_t best_den, uint32_t rounds, double time_ms, double h2d_ms) {
    char buf[1024];
    snprintf(buf, sizeof buf, "%s,%s,%u,%u,%llu,%llu,%.6f,%u,%.3f,%.3f,%.3f", impl, graph, n, m,
             (unsigned long long)best_num, (unsigned long long)best_den, density_value(best_num, best_den), rounds,
             time_ms, h2d_ms, time_ms + h2d_ms);
    return buf;
}

// ---- host-side GPU peeling loop, independent of CUDA through a "backend" -------------------------------------
// The loop below is the code that gpu_densest.cu runs, and the emulator test runs the SAME code with a CPU
// backend that drives the real kernels through emu/cuda_shim.h. A backend B provides:
//   B.reset_state()                 alive[] = 1, removed_round[] = UINT32_MAX (device side)
//   B.begin_round()                 zero tri[] and the two counters (cudaMemset)
//   B.run_tc()                      launch tc_per_vertex_alive
//   B.run_mark(nalive,num,den,rnd)  launch mark_peel (reads T from the device counter, so no host sync between kernels)
//   B.read_stats(T, removed)        copy the two uint64 counters {T, removed} back = the ONE host sync per round
//   B.kernel_ms()                   (optional diagnostics) kernel time of the last round, or < 0
// Deliberately simple design: best-density tracking and the alive count live on the host, costing one tiny
// device-to-host copy (16 bytes) plus a pipeline drain every round - about a few tens of microseconds each on a
// PCIe GPU - times the number of rounds (O(log n / eps)). A fully device-side loop would avoid it; not done.
// T_full = triangles of the whole graph (from the serial CPU counter); it is used for the overflow check and as a
// cross-check of the device count in round 0. The caller downloads removed_round from the device afterwards.
template <class B>
inline PeelResult batch_peel_loop(uint32_t n, uint64_t T_full, uint64_t num, uint64_t den, B& be,
                                  bool verbose = false, FILE* diag = nullptr) {
    PeelResult r;
    std::string why;
    if (!peel_arith_ok(n, T_full, num, den, &why)) { r.ok = false; r.error = why; return r; }
    be.reset_state();
    uint64_t nalive = n;
    for (uint32_t round = 0; nalive > 0; round++) {
        be.begin_round();
        be.run_tc();
        be.run_mark(nalive, num, den, round);
        uint64_t T = 0, removed = 0;
        be.read_stats(T, removed);
        if (round == 0 && T != T_full) { r.ok = false; r.error = "device triangle count in round 0 differs from CPU count"; return r; }
        if (verbose && diag)
            fprintf(diag, "round %4u  alive=%10llu  T=%12llu  removed=%10llu  kernel_ms=%.3f\n", round,
                    (unsigned long long)nalive, (unsigned long long)T, (unsigned long long)removed, be.kernel_ms());
        if (peel_improves(T, nalive, r.best_num, r.best_den)) { r.best_num = T; r.best_den = nalive; r.best_round = round; }
        if (removed == 0 || removed > nalive) { r.ok = false; r.error = "bad removed count from device (0 or more than alive)"; return r; }
        nalive -= removed;
        r.rounds = round + 1;
    }
    return r;
}
