// dsg.h - paper algorithms for the TRIANGLE-DENSITY densest subgraph (CPU reference; the GPU only replaces the
// core decomposition, see gpu_dsg.cu). Psi = triangle (h = 3, |V_Psi| = 3) throughout.
//
//   Fang, Yu, Cheng, Lakshmanan, Lin: "Efficient Algorithms for Densest Subgraph Discovery", PVLDB 2019 ("DSD paper")
//     Alg 1  Exact       flow network + binary search on alpha           -> exact_densest()
//     Alg 3  (k,Psi)-core decomposition                                  -> core_decomposition()
//     Thm 1  k/|V_Psi| <= rho(R_k) <= kmax,  Lemma 5 rho_opt <= kmax,  Lemma 7 CDS is in the ceil(rho_opt)-core
//     Alg 4  CoreExact   (Pruning1/2/3 + shrinking flow network)         -> core_exact()
//   Boob, Gao, Peng, Sawlani, Tsourakakis, Wang, Wang: "Flowless: Extracting Densest Subgraphs Without Flow
//     Computations", WWW 2020 ("Flowless paper")
//     Alg 1  Greedy (Charikar)   Alg 2  Greedy++                         -> greedy(), greedy_pp()
//   Greedy++ is stated for edges; here every TRIANGLE is a hyperedge: deg_H(u) becomes tri_H(u), the number of
//   triangles of the current subgraph H that contain u. Nothing else changes.
//
// DENSITY. rho(S) = triangles(S) / |S|. All decisions use exact integers: densities are fractions (num, den) compared
// by cross multiplication in unsigned __int128 (Dens / dens_cmp). The only floating point is density_value() for display.
// The empty answer ("the graph has no triangle") is Dens{0,0}, which compares equal to every 0/x.
//
// FLOW ARITHMETIC. Binary search on alpha is done on the integer grid alpha = a / Q with Q = 2 n (n-1) (n = vertices of
// the network). All capacities are multiplied by Q, so they are integers: s->v : Q*tri(v), v->t : 3a, v->psi : Q,
// psi->v : infinity. Termination (exact): two different densities t1/s1 != t2/s2 with s1,s2 <= n differ by at least
// 1/(n(n-1)); the search stops when hi-lo < 2 grid units, i.e. u - l < 1/(n(n-1)) (the paper's stopping rule).
// build() refuses networks whose total capacity could reach 2^59 (int64 flow).
#pragma once
#include <array>
#include <functional>
#include <numeric>
#include <string>
#include "densest.h"

typedef unsigned __int128 u128;
typedef int64_t i64;

// ----------------------------------------------------------------------------------------------------------
// exact rationals
// ----------------------------------------------------------------------------------------------------------
struct Dens {
    uint64_t num = 0, den = 0;      // triangles / vertices; den == 0 means "no subgraph", density 0
};
inline int dens_cmp(Dens a, Dens b) {            // -1, 0, +1, exact
    if (a.den == 0) a = Dens{0, 1};
    if (b.den == 0) b = Dens{0, 1};
    const u128 l = (u128)a.num * b.den, r = (u128)b.num * a.den;
    return l < r ? -1 : (l > r ? 1 : 0);
}
inline bool dens_gt(Dens a, Dens b) { return dens_cmp(a, b) > 0; }
inline uint64_t dens_ceil(Dens d) { return d.den == 0 ? 0 : (d.num + d.den - 1) / d.den; }
inline double dens_value(Dens d) { return density_value(d.num, d.den); }

struct DsgResult {
    uint64_t tri = 0, verts = 0;          // best subgraph: triangles / vertices (0/0 = graph without triangles)
    std::vector<uint32_t> set;            // its vertices (ids of the graph that was passed in), sorted ascending
    std::vector<Dens> per_pass;           // greedy_pp only: best density after pass 1, 2, ..., T
    uint32_t flow_calls = 0;              // number of max-flow computations
    uint32_t components_searched = 0;     // core_exact: components that passed lines 7-9 and entered the binary search
    uint32_t shrinks = 0;                 // core_exact: times line 17 rebuilt the network on a higher core
    bool ok = true;
    std::string error;
    Dens dens() const { return Dens{tri, verts}; }
};

// ----------------------------------------------------------------------------------------------------------
// triangle enumeration on the oriented CSR
// ----------------------------------------------------------------------------------------------------------
// Calls f(u, v, w, e_uv, e_uw, e_vw) once per triangle; e_xy is the index into g.col of the oriented edge x->y.
// (u,v) is oriented edge e; w is a common out-neighbour of u and v, found at position i of u's list and j of v's list.
template <class F>
inline void for_each_triangle(const Graph& g, F&& f) {
    for (uint32_t e = 0; e < g.m; e++) {
        const uint32_t u = g.src[e], v = g.col[e];
        uint32_t i = g.rowptr[u], iend = g.rowptr[u + 1], j = g.rowptr[v], jend = g.rowptr[v + 1];
        while (i < iend && j < jend) {
            const uint32_t a = g.col[i], b = g.col[j];
            if (a < b) i++;
            else if (a > b) j++;
            else { f(u, v, a, e, i, j); i++; j++; }
        }
    }
}

// Per-vertex triangle degree deg_G(v, Psi) and the total number of triangles (DSD paper Def. 3).
inline uint64_t triangle_degrees(const Graph& g, std::vector<uint64_t>& deg) {
    deg.assign(g.n, 0);
    uint64_t T = 0;
    for_each_triangle(g, [&](uint32_t u, uint32_t v, uint32_t w, uint32_t, uint32_t, uint32_t) { T++; deg[u]++; deg[v]++; deg[w]++; });
    return T;
}

// Triangles with all three corners in mask[].
inline uint64_t count_triangles_in(const Graph& g, const std::vector<uint8_t>& mask) {
    uint64_t T = 0;
    for_each_triangle(g, [&](uint32_t u, uint32_t v, uint32_t w, uint32_t, uint32_t, uint32_t) { T += mask[u] && mask[v] && mask[w]; });
    return T;
}

// ----------------------------------------------------------------------------------------------------------
// induced subgraph
// ----------------------------------------------------------------------------------------------------------
// New compact Graph on the vertices with keep[v] != 0. New ids follow the order of the old ids (old_id[new] ascending).
// It is a complete, valid oriented CSR: the surviving undirected edges go through build_graph_from_edges (the same
// construction as load_graph), so the orientation uses the degrees INSIDE the subgraph, rows are sorted, src[] and
// max_out_degree are filled.
inline void induced_subgraph(const Graph& g, const std::vector<uint8_t>& keep, Graph& out, std::vector<uint32_t>& old_id) {
    std::vector<uint32_t> new_id(g.n, UINT32_MAX);
    old_id.clear();
    for (uint32_t v = 0; v < g.n; v++)
        if (keep[v]) { new_id[v] = (uint32_t)old_id.size(); old_id.push_back(v); }
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for (uint32_t e = 0; e < g.m; e++) {
        const uint32_t a = g.src[e], b = g.col[e];
        if (keep[a] && keep[b]) edges.emplace_back(new_id[a], new_id[b]);
    }
    build_graph_from_edges((uint32_t)old_id.size(), edges, out);
}

// ----------------------------------------------------------------------------------------------------------
// (k, Psi)-core decomposition: DSD paper Alg 3
// ----------------------------------------------------------------------------------------------------------
// Repeatedly remove the vertex v with the smallest current triangle degree; core[v] = that degree (Alg 3 line 5).
// For every triangle {v,a,b} that still has both other corners alive, a and b lose one triangle, but ONLY if their
// degree is larger than deg(v) (lines 8-9): degrees never fall below the degree of the vertex being removed, which is
// what makes core[] the max over time of the minimum degree = the order of the highest (k,Psi)-core containing v.
// "Sort / re-sort" (lines 3, 11) is a lazy-deletion min-heap on (degree, id) instead of the paper's bin sort, so
// the cost is O(T log n) plus the triangle enumerations. Returns false if a core number does not fit uint32.
inline bool core_decomposition(const Graph& g, std::vector<uint32_t>& core, uint32_t* kmax_out = nullptr) {
    std::vector<uint32_t> urow, ucol;
    build_undirected_csr(g, urow, ucol);
    std::vector<uint64_t> deg;
    triangle_degrees(g, deg);
    core.assign(g.n, 0);
    std::vector<uint8_t> alive(g.n, 1), is_touched(g.n, 0);
    typedef std::pair<uint64_t, uint32_t> Entry;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    for (uint32_t v = 0; v < g.n; v++) heap.emplace(deg[v], v);
    uint32_t kmax = 0;
    std::vector<uint32_t> touched;
    for (uint32_t done = 0; done < g.n;) {
        const Entry top = heap.top(); heap.pop();
        const uint32_t v = top.second;
        if (!alive[v] || deg[v] != top.first) continue;           // stale heap entry
        if (deg[v] > UINT32_MAX) return false;
        core[v] = (uint32_t)deg[v];                               // Alg 3 line 5
        kmax = std::max(kmax, core[v]);
        alive[v] = 0;
        done++;
        touched.clear();
        const uint32_t* nv = ucol.data() + urow[v];
        const uint32_t dv = urow[v + 1] - urow[v];
        for (uint32_t k = 0; k < dv; k++) {
            const uint32_t a = nv[k];
            if (!alive[a]) continue;
            for_common_above(nv, dv, ucol.data() + urow[a], urow[a + 1] - urow[a], a, [&](uint32_t b) {
                if (!alive[b]) return;                            // triangle {v,a,b} already destroyed
                const uint32_t us[2] = {a, b};
                for (uint32_t u : us)                             // Alg 3 lines 6-9
                    if (deg[u] > core[v]) {
                        deg[u]--;
                        if (!is_touched[u]) { is_touched[u] = 1; touched.push_back(u); }
                    }
            });
        }
        for (uint32_t x : touched) { is_touched[x] = 0; heap.emplace(deg[x], x); }
    }
    if (kmax_out) *kmax_out = kmax;
    return true;
}

// Triangles / vertices of {v : core[v] >= k}, the (k,Psi)-core.
inline Dens core_level_density(const Graph& g, const std::vector<uint32_t>& core, uint32_t k) {
    std::vector<uint8_t> in(g.n);
    uint64_t cnt = 0;
    for (uint32_t v = 0; v < g.n; v++) { in[v] = core[v] >= k; cnt += in[v]; }
    return Dens{count_triangles_in(g, in), cnt};
}

// ----------------------------------------------------------------------------------------------------------
// Greedy (Flowless Alg 1) and Greedy++ (Flowless Alg 2), triangle version
// ----------------------------------------------------------------------------------------------------------
// Greedy++, T passes. Load vector l (Alg 2 line 2, zero). Pass i: H = G; while H nonempty: pick u in H minimising
// l^(i-1)_u + tri_H(u) (line 6), set l^(i)_u = l^(i-1)_u + tri_H(u) (line 7), delete u (line 8), and remember H if it is
// denser than the best so far (lines 9-10; we evaluate H before every deletion, which visits the same subgraphs: G itself
// is Alg 2 line 1 and the empty graph never wins). Pass 1 has l = 0 and is exactly Charikar's Greedy (Alg 1 / DSD paper
// PeelApp), ties broken by smaller id.
// Because a deleted vertex is never looked at again within the pass, writing l^(i)_u into a second array (as the paper
// does) or in place would give the same result; a second array keeps it identical to the pseudocode.
// Lazy-deletion min-heap on (l_u + tri_H(u), id): the key of a vertex only ever DEcreases inside a pass (its triangle count
// drops), so the newest heap entry of an alive vertex is the valid one. per_pass[i-1] = best density after pass i.
// Self-check: the triangles enumerated when deleting u must equal tri_H(u).
inline DsgResult greedy_pp(const Graph& g, uint32_t passes) {
    DsgResult r;
    std::vector<uint32_t> urow, ucol;
    build_undirected_csr(g, urow, ucol);
    std::vector<uint64_t> tri0;
    const uint64_t T0 = triangle_degrees(g, tri0);
    std::vector<uint64_t> load(g.n, 0), next_load(g.n, 0), cur(g.n);
    std::vector<uint8_t> alive(g.n), is_touched(g.n, 0);
    std::vector<uint32_t> order, best_order, touched;
    size_t best_cut = 0;
    Dens best{0, 0};
    typedef std::pair<uint64_t, uint32_t> Entry;
    for (uint32_t pass = 1; pass <= passes; pass++) {
        cur = tri0;
        alive.assign(g.n, 1);
        uint64_t Tcur = T0, nalive = g.n;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
        for (uint32_t v = 0; v < g.n; v++) heap.emplace(load[v] + cur[v], v);
        order.clear();
        bool improved = false;
        size_t cut = 0;
        while (nalive > 0) {
            if (dens_gt(Dens{Tcur, nalive}, best)) { best = Dens{Tcur, nalive}; improved = true; cut = order.size(); }
            const Entry top = heap.top(); heap.pop();
            const uint32_t v = top.second;
            if (!alive[v] || load[v] + cur[v] != top.first) continue;     // stale entry
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
            if (!r.ok || found != cur[v]) { r.ok = false; r.error = "greedy_pp: triangle bookkeeping inconsistent"; return r; }
            next_load[v] = load[v] + cur[v];                              // Alg 2 line 7
            alive[v] = 0;
            order.push_back(v);
            Tcur -= cur[v];
            nalive--;
            for (uint32_t x : touched) { is_touched[x] = 0; heap.emplace(load[x] + cur[x], x); }
        }
        if (improved) { best_order = order; best_cut = cut; }
        load = next_load;
        r.per_pass.push_back(best);
    }
    r.tri = best.num;
    r.verts = best.den;
    r.set.assign(best_order.begin() + best_cut, best_order.end());
    if (best.den == 0) r.set.clear();
    std::sort(r.set.begin(), r.set.end());
    return r;
}
inline DsgResult greedy(const Graph& g) { return greedy_pp(g, 1); }

// ----------------------------------------------------------------------------------------------------------
// Dinic max-flow (iterative, int64 capacities)
// ----------------------------------------------------------------------------------------------------------
struct Dinic {
    static constexpr i64 INF = (i64)1 << 60;
    int N = 0;
    std::vector<int> head, nxt, to, level, it;
    std::vector<i64> cap;
    void init(int n) { N = n; head.assign(n, -1); nxt.clear(); to.clear(); cap.clear(); }
    int add_edge(int u, int v, i64 c) {                 // returns the id of the forward arc; arc id^1 is its reverse
        const int id = (int)to.size();
        to.push_back(v); cap.push_back(c); nxt.push_back(head[u]); head[u] = id;
        to.push_back(u); cap.push_back(0); nxt.push_back(head[v]); head[v] = id + 1;
        return id;
    }
    bool bfs(int s, int t) {                            // level graph; leaves level[] = reachability when it returns false
        level.assign(N, -1);
        std::vector<int> q;
        q.push_back(s); level[s] = 0;
        for (size_t qi = 0; qi < q.size(); qi++) {
            const int u = q[qi];
            for (int e = head[u]; e != -1; e = nxt[e])
                if (cap[e] > 0 && level[to[e]] < 0) { level[to[e]] = level[u] + 1; q.push_back(to[e]); }
        }
        return level[t] >= 0;
    }
    // Blocking flow with current-arc pointers and an explicit path stack (the level graph can be thousands deep).
    i64 maxflow(int s, int t) {
        i64 flow = 0;
        std::vector<int> path;
        while (bfs(s, t)) {
            it = head;
            path.clear();
            int u = s;
            for (;;) {
                if (u == t) {
                    i64 f = INF;
                    for (int e : path) f = std::min(f, cap[e]);
                    for (int e : path) { cap[e] -= f; cap[e ^ 1] += f; }
                    flow += f;
                    size_t k = 0;
                    while (cap[path[k]] > 0) k++;       // first saturated arc: restart from its tail
                    u = to[path[k] ^ 1];
                    path.resize(k);
                    continue;
                }
                int& e = it[u];
                while (e != -1 && !(cap[e] > 0 && level[to[e]] == level[u] + 1)) e = nxt[e];
                if (e != -1) { path.push_back(e); u = to[e]; }
                else {                                  // dead end: retreat and never enter u again in this phase
                    if (u == s) break;
                    level[u] = -1;
                    const int pe = path.back(); path.pop_back();
                    u = to[pe ^ 1];
                    it[u] = nxt[it[u]];
                }
            }
        }
        return flow;
    }
};

// ----------------------------------------------------------------------------------------------------------
// flow network of DSD paper Alg 1 lines 5-15 for Psi = triangle
// ----------------------------------------------------------------------------------------------------------
// Nodes: s, t, one node per vertex of H, and one node per (h-1)-clique = per EDGE psi of H (Lambda in the paper).
// (Only edges that lie in at least one triangle get a node; the others would have no incoming arc and never matter.)
//   s -> v     capacity deg(v, Psi)                    (line 7)   [x Q]
//   v -> t     capacity alpha * |V_Psi| = 3 alpha      (line 8)   [3a, alpha = a/Q]
//   psi -> v   capacity infinity for both ends of psi  (line 11)
//   v -> psi   capacity 1 if psi + v is a triangle     (line 15)  [x Q]
// Why this finds a subgraph denser than alpha: let S be the vertex set on the source side. For every triangle with k corners
// in S the cut pays: k=3: 0, k=2: 2 (two arcs v->psi from S into a sink-side psi) + 1 (s->w for the corner outside S) = 3,
// k=1: 1+2 = 3, k=0: 3 (three s->v arcs) - i.e. 3 per triangle not inside S. So cut(S) = 3 (T - T(S)) + 3 alpha |S| and
// cut(S) < 3T <=> T(S) > alpha |S| <=> rho(S) > alpha. The trivial cut S={s} costs 3T. We take the MINIMAL source side
// (reachable from s in the residual graph): it is {s} exactly when no subgraph is denser than alpha (paper lines 17-18).
// (The task text said "one node per triangle"; the paper's Alg 1 uses the (h-1)-cliques, i.e. edges, for h = 3. That is
// what is implemented; it is also smaller: 3 arcs per triangle instead of 3 + one node.)
struct TriNet {
    Dinic D;
    uint32_t n = 0;
    uint64_t Q = 1;
    std::vector<std::array<uint32_t, 3>> tris;
    std::vector<uint64_t> deg;
    std::vector<int> sink_arc;       // arc id of v->t, -1 for vertices in no triangle
    std::vector<i64> cap0;

    bool build(const Graph& H, uint64_t Q_, std::string* err) {
        n = H.n; Q = Q_;
        tris.clear();
        deg.assign(n, 0);
        std::vector<int> enode(H.m, -1);
        std::vector<std::array<uint32_t, 3>> tedges;
        for_each_triangle(H, [&](uint32_t u, uint32_t v, uint32_t w, uint32_t euv, uint32_t euw, uint32_t evw) {
            tris.push_back({u, v, w});
            tedges.push_back({euv, euw, evw});
            deg[u]++; deg[v]++; deg[w]++;
        });
        if ((u128)tris.size() * 3 * Q >= ((u128)1 << 59)) { if (err) *err = "flow capacities would exceed 2^59"; return false; }
        int cnt = 2 + (int)n;
        for (auto& te : tedges) for (uint32_t e : te) if (enode[e] < 0) enode[e] = cnt++;
        D.init(cnt);
        sink_arc.assign(n, -1);
        for (uint32_t v = 0; v < n; v++) {
            if (!deg[v]) continue;
            D.add_edge(0, 2 + (int)v, (i64)(Q * deg[v]));                     // line 7
            sink_arc[v] = D.add_edge(2 + (int)v, 1, 0);                        // line 8, capacity set per alpha
        }
        std::vector<uint8_t> made(H.m, 0);
        for (size_t i = 0; i < tris.size(); i++) {
            const auto& t = tris[i];
            const auto& te = tedges[i];
            // edge te[0] = (u,v) closes with w, te[1] = (u,w) with v, te[2] = (v,w) with u
            const uint32_t ends[3][2] = {{t[0], t[1]}, {t[0], t[2]}, {t[1], t[2]}};
            const uint32_t third[3] = {t[2], t[1], t[0]};
            for (int k = 0; k < 3; k++) {
                const int pn = enode[te[k]];
                if (!made[te[k]]) {                                            // line 11
                    made[te[k]] = 1;
                    D.add_edge(pn, 2 + (int)ends[k][0], Dinic::INF);
                    D.add_edge(pn, 2 + (int)ends[k][1], Dinic::INF);
                }
                D.add_edge(2 + (int)third[k], pn, (i64)Q);                     // line 15
            }
        }
        cap0 = D.cap;
        return true;
    }
    // Vertices (ids of H) on the minimal source side for alpha = a / Q; empty = no subgraph denser than alpha.
    std::vector<uint32_t> solve(i64 a) {
        D.cap = cap0;
        for (uint32_t v = 0; v < n; v++) if (sink_arc[v] >= 0) D.cap[sink_arc[v]] = 3 * a;
        D.maxflow(0, 1);
        std::vector<uint32_t> S;
        for (uint32_t v = 0; v < n; v++) if (deg[v] && D.level[2 + v] >= 0) S.push_back(v);
        return S;
    }
    uint64_t count_in(const std::vector<uint32_t>& S) const {
        std::vector<uint8_t> in(n, 0);
        for (uint32_t v : S) in[v] = 1;
        uint64_t c = 0;
        for (auto& t : tris) c += in[t[0]] && in[t[1]] && in[t[2]];
        return c;
    }
};

// ----------------------------------------------------------------------------------------------------------
// Exact (DSD paper Alg 1)
// ----------------------------------------------------------------------------------------------------------
// l = 0, u = max_v deg(v, Psi) (line 1). While u - l >= 1/(n(n-1)): alpha = (l+u)/2; min cut; S = {s}: u = alpha, else
// l = alpha and D = the vertex set of S (lines 3-18). Why the final D is optimal: D has density > l (it beat the last
// successful alpha), the optimum is <= u, and u - l < 1/(n(n-1)); two distinct densities with denominators <= n differ
// by at least 1/(n(n-1)), so rho(D) = rho_opt. Integer grid: see the header comment (Q = 2n(n-1), loop until hi-lo < 2).
inline DsgResult exact_densest(const Graph& g) {
    DsgResult r;
    std::vector<uint64_t> deg;
    const uint64_t T = triangle_degrees(g, deg);
    if (T == 0) return r;
    const uint64_t maxdeg = *std::max_element(deg.begin(), deg.end());
    const uint64_t Q = 2ull * g.n * (g.n - 1);
    TriNet net;
    if (!net.build(g, Q, &r.error)) { r.ok = false; return r; }
    i64 lo = 0, hi = (i64)(maxdeg * Q);
    Dens best{0, 0};
    while (hi - lo >= 2) {
        const i64 mid = lo + (hi - lo) / 2;
        std::vector<uint32_t> S = net.solve(mid);
        r.flow_calls++;
        if (S.empty()) { hi = mid; continue; }
        lo = mid;
        const Dens d{net.count_in(S), S.size()};
        if ((u128)d.num * Q <= (u128)mid * d.den) { r.ok = false; r.error = "min cut returned a set that is not denser than alpha"; return r; }
        if (dens_gt(d, best)) { best = d; r.set = S; }
    }
    r.tri = best.num; r.verts = best.den;
    return r;
}

// ----------------------------------------------------------------------------------------------------------
// CoreExact (DSD paper Alg 4)
// ----------------------------------------------------------------------------------------------------------
// INPUT core[] = the (k,Psi)-core numbers of g (from core_decomposition() or from the GPU). Steps:
//  1-2. Pruning1: the residual graphs of the peeling are the suffixes of the vertex order sorted by (core, id) (every
//       (k,Psi)-core is such a suffix). rho' = best density among them (a real subgraph, so rho' <= rho_opt) and by Lemma 7
//       the CDS lies in the k'-core, k' = ceil(rho'). The kmax-core is one of the suffixes, so rho' >= kmax/3 (Thm 1).
//       Pruning2: rho'' = max(rho', densities of the connected components of the k'-core); k'' = max(k', ceil(rho'')).
//  3-4. l = rho'' (kept as "the density of the best subgraph found so far", an exact fraction), u = kmax, C = connected
//       components of the k''-core.
//  5-20. For each component: line 6, restrict to the ceil(l)-core; build the network (line 7); if alpha = l already has
//       no denser subgraph (S = {s}) skip the component (lines 8-9); otherwise binary search (lines 10-19) with the stopping
//       rule of Pruning3 (|V_C|(|V_C|-1), here the grid Q = 2 |V_C|(|V_C|-1)); whenever a feasible alpha raises l above the
//       current core level, the network is rebuilt on the (ceil(alpha))-core of C (line 17 "remove some vertices from C").
// Points where this differs from / interprets the paper (all keep exactness; see the tests):
//   * u is reset to kmax for EVERY component, l is not. The paper sets u once (line 3); a later component would then
//     inherit an upper bound that was only proved for earlier components.
//   * l is the exact density of the best subgraph found so far (>= the paper's alpha), the grid start is floor(l*Q)/Q <= l.
//   * the initial check (lines 7-9) also records the subgraph it finds as a candidate (the paper keeps only U from line 19);
//     without that, a component whose optimum lies in the last tiny interval would not be returned.
//   * (k,Psi)-cores of a component C of the k''-core are taken from the INPUT core numbers: for k >= k'' the k-core of
//     G[C] is {v in C : core_G[v] >= k} (the k-core of G inside C is a subgraph of C with minimum degree >= k, and every such
//     subgraph of C is inside the k-core of G), so no re-decomposition is needed.
// Why each prune is safe: any densest subgraph D has tri_D(v) >= rho_opt for every v in D (Lemma 4), hence core[v] >=
// ceil(rho_opt) (Lemma 7). Any real subgraph gives rho <= rho_opt, so cores of ceil(l) and ceil(alpha) are supersets of
// the ceil(rho_opt)-core.
//
// OPTIONAL SEED (stage 6b): `seed` = a result of another algorithm on the SAME graph (e.g. Greedy++), i.e. a real vertex
// set with its triangle count. It is checked here (set size, no duplicates, triangle count recomputed) and, if denser than
// the lower bound from Pruning1/2, replaces it as l. A better l gives a higher ceil(l)-core and, above all, a start
// alpha that is already at the optimum, so the very first max-flow (line 7-8) usually answers "nothing denser" (line 9)
// and the whole binary search is skipped: one flow call instead of ~log2(kmax * Q). Exactness does not depend on the seed
// being good, only on it being a real subgraph (which is verified).
inline DsgResult core_exact(const Graph& g, const std::vector<uint32_t>& core, bool use_pruning12 = true,
                            const DsgResult* seed = nullptr) {
    DsgResult r;
    if (core.size() != g.n) { r.ok = false; r.error = "core[] has the wrong size"; return r; }
    const uint32_t kmax = g.n ? *std::max_element(core.begin(), core.end()) : 0;
    if (kmax == 0) return r;                                                    // no triangle at all
    std::vector<uint32_t> urow, ucol;
    build_undirected_csr(g, urow, ucol);

    Dens best{0, 0};
    std::vector<uint32_t> best_set;

    // connected components of {core >= k}; returns the label per vertex (UINT32_MAX = not in the core) and their count
    auto components = [&](uint32_t k, std::vector<uint32_t>& label) {
        label.assign(g.n, UINT32_MAX);
        uint32_t nc = 0;
        std::vector<uint32_t> stack;
        for (uint32_t s = 0; s < g.n; s++) {
            if (core[s] < k || label[s] != UINT32_MAX) continue;
            label[s] = nc; stack.push_back(s);
            while (!stack.empty()) {
                const uint32_t x = stack.back(); stack.pop_back();
                for (uint32_t p = urow[x]; p < urow[x + 1]; p++) {
                    const uint32_t y = ucol[p];
                    if (core[y] >= k && label[y] == UINT32_MAX) { label[y] = nc; stack.push_back(y); }
                }
            }
            nc++;
        }
        return nc;
    };

    std::vector<uint32_t> label;
    uint64_t k2 = 1;                                                            // k''
    if (use_pruning12) {
        // --- Pruning1: residual graphs of the peeling = suffixes of the (core, id) order ---
        std::vector<uint32_t> order(g.n), pos(g.n);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return core[a] != core[b] ? core[a] < core[b] : a < b; });
        for (uint32_t i = 0; i < g.n; i++) pos[order[i]] = i;
        std::vector<uint64_t> cnt(g.n, 0);        // cnt[i] = triangles whose earliest vertex in the order is order[i]
        for_each_triangle(g, [&](uint32_t u, uint32_t v, uint32_t w, uint32_t, uint32_t, uint32_t) { cnt[std::min({pos[u], pos[v], pos[w]})]++; });
        {
            uint64_t suffix = 0;
            size_t best_i = g.n;
            for (uint32_t i = g.n; i-- > 0;) {
                suffix += cnt[i];
                if (dens_gt(Dens{suffix, g.n - i}, best)) { best = Dens{suffix, g.n - i}; best_i = i; }
            }
            best_set.assign(order.begin() + best_i, order.end());
        }
        const uint64_t k1 = dens_ceil(best);
        if (k1 > kmax) { r.ok = false; r.error = "inconsistent core numbers (ceil(rho') > kmax)"; return r; }

        // --- Pruning2: components of the k'-core ---
        const uint32_t nc1 = components((uint32_t)k1, label);
        std::vector<uint64_t> ct(nc1, 0), cv(nc1, 0);
        for (uint32_t v = 0; v < g.n; v++) if (label[v] != UINT32_MAX) cv[label[v]]++;
        for_each_triangle(g, [&](uint32_t u, uint32_t v, uint32_t w, uint32_t, uint32_t, uint32_t) {
            if (label[u] != UINT32_MAX && label[v] != UINT32_MAX && label[w] != UINT32_MAX) ct[label[u]]++;
        });
        for (uint32_t c = 0; c < nc1; c++)
            if (dens_gt(Dens{ct[c], cv[c]}, best)) {
                best = Dens{ct[c], cv[c]};
                best_set.clear();
                for (uint32_t v = 0; v < g.n; v++) if (label[v] == c) best_set.push_back(v);
            }
        k2 = std::max<uint64_t>(k1, dens_ceil(best));
        if (k2 > kmax) { r.ok = false; r.error = "inconsistent core numbers (ceil(rho'') > kmax)"; return r; }
    } else {
        // TEST HOOK (use_pruning12 = false): start from the weakest real lower bound, one triangle (density 1/3, so k'' = 1),
        // instead of Pruning1/2. The result must be the same; the binary search then has to climb over many integer core levels,
        // which exercises the line-17 network shrinking and the multi-component loop much harder than the real start does.
        bool found = false;
        for_each_triangle(g, [&](uint32_t u, uint32_t v, uint32_t w, uint32_t, uint32_t, uint32_t) {
            if (found) return;
            found = true;
            best = Dens{1, 3};
            best_set = {u, v, w};
        });
    }

    if (seed && seed->ok && seed->verts > 0) {
        // verify the seed is a real subgraph with the claimed numbers before trusting it as a lower bound
        std::vector<uint8_t> in(g.n, 0);
        bool okseed = seed->set.size() == seed->verts;
        for (uint32_t v : seed->set) { if (v >= g.n || in[v]) { okseed = false; break; } in[v] = 1; }
        if (!okseed || count_triangles_in(g, in) != seed->tri) { r.ok = false; r.error = "seed is not a valid subgraph of the graph"; return r; }
        const Dens sd{seed->tri, seed->verts};
        if (dens_gt(sd, best)) { best = sd; best_set = seed->set; }
        k2 = std::max<uint64_t>(k2, dens_ceil(best));
        if (k2 > kmax) { r.ok = false; r.error = "inconsistent seed (ceil(rho) > kmax)"; return r; }
    }

    // --- lines 4-20: one component at a time ---
    const uint32_t nc = components((uint32_t)k2, label);
    for (uint32_t c = 0; c < nc; c++) {
        // line 6: C <- C ∩ (ceil(l))-core, with l = the density of the best subgraph so far
        uint64_t kc = std::max<uint64_t>(k2, dens_ceil(best));
        if (kc > kmax) break;                                                  // nothing can beat l (rho <= kmax)
        std::vector<uint8_t> mask(g.n, 0);
        for (uint32_t v = 0; v < g.n; v++) mask[v] = label[v] == c && core[v] >= kc;
        Graph H;
        std::vector<uint32_t> old_id;                                          // H id -> id in g
        induced_subgraph(g, mask, H, old_id);
        if (H.n < 3) continue;
        TriNet net;
        std::string why;
        // the grid is fixed from the size of C after line 6 (Pruning3); shrinking later only makes the true n smaller
        const uint64_t Q = 2ull * H.n * (H.n - 1);
        if (!net.build(H, Q, &why)) { r.ok = false; r.error = why; return r; }
        if (net.tris.empty()) continue;
        // Start alpha = ceil(l*Q)/Q, just above l (by < 1/Q). If the flow at this alpha finds nothing denser (line 9), l is
        // optimal for this component: two different subgraph densities with <= H.n vertices differ by >= 1/(n(n-1)) = 2/Q,
        // so nothing can lie in (l, ceil(l*Q)/Q]. If it finds a denser set, alpha is feasible and the search continues.
        i64 lo = best.den ? (i64)(((u128)best.num * Q + best.den - 1) / best.den) : 0;
        i64 hi = (i64)((u128)kmax * Q);                                        // u = kmax, reset per component
        if (lo >= hi) continue;

        auto consider = [&](const std::vector<uint32_t>& S, i64 alpha_a) -> bool {
            const Dens d{net.count_in(S), S.size()};
            if ((u128)d.num * Q <= (u128)alpha_a * d.den) { r.ok = false; r.error = "min cut returned a set that is not denser than alpha"; return false; }
            if (dens_gt(d, best)) {
                best = d;
                best_set.clear();
                for (uint32_t v : S) best_set.push_back(old_id[v]);
            }
            return true;
        };
        std::vector<uint32_t> S = net.solve(lo);                               // lines 7-8
        r.flow_calls++;
        if (S.empty()) continue;                                               // line 9
        if (!consider(S, lo)) return r;
        r.components_searched++;
        while (hi - lo >= 2) {                                                 // line 10
            const i64 mid = lo + (hi - lo) / 2;                                // line 11
            S = net.solve(mid);                                                // lines 12-13
            r.flow_calls++;
            if (S.empty()) { hi = mid; continue; }                             // lines 14-15
            lo = mid;                                                          // line 18
            if (!consider(S, mid)) return r;                                   // lines 19-20
            const uint64_t knew = ((uint64_t)mid + Q - 1) / Q;                 // ceil(alpha)
            if (knew > kc) {                                                   // line 17: shrink C to the ceil(alpha)-core
                kc = knew;
                r.shrinks++;
                if (kc > kmax) break;
                std::vector<uint8_t> m2(H.n, 0);
                for (uint32_t v = 0; v < H.n; v++) m2[v] = core[old_id[v]] >= kc;
                Graph H2;
                std::vector<uint32_t> map2;
                induced_subgraph(H, m2, H2, map2);
                std::vector<uint32_t> old2(map2.size());
                for (size_t i = 0; i < map2.size(); i++) old2[i] = old_id[map2[i]];
                if (H2.n < 3) { r.ok = false; r.error = "shrunk network is empty although a denser subgraph exists"; return r; }
                if (!net.build(H2, Q, &why)) { r.ok = false; r.error = why; return r; }
                if (net.tris.empty()) { r.ok = false; r.error = "shrunk network has no triangle although a denser subgraph exists"; return r; }
                H = std::move(H2);
                old_id = std::move(old2);
            }
        }
    }
    std::sort(best_set.begin(), best_set.end());
    r.tri = best.num; r.verts = best.den; r.set = best_set;
    return r;
}

// ----------------------------------------------------------------------------------------------------------
// pruning used by the GPU pipeline (and tested with CPU core numbers)
// ----------------------------------------------------------------------------------------------------------
// Containment theorem (DSD paper Lemma 7 / Section 6.1): every densest subgraph lies in the ceil(rho_opt)-core. Any real
// subgraph has density lb <= rho_opt, hence ceil(lb) <= ceil(rho_opt) and the ceil(lb)-core contains every densest
// subgraph. Cores nest, so we may discard all vertices with core[v] < ceil(lb). Core numbers of the pruned graph G' equal
// the original ones for its vertices: for j >= k the j-core of G' = the j-core of G (the j-core of G is inside G'), and
// G' itself has minimum triangle degree >= k, so no vertex of G' can have core < k. So core[old_id[i]] IS the core
// number of vertex i of G' and no second decomposition is needed.
inline void prune_to_core(const Graph& g, const std::vector<uint32_t>& core, uint64_t k, Graph& out,
                          std::vector<uint32_t>& old_id, std::vector<uint32_t>& core_out) {
    std::vector<uint8_t> keep(g.n);
    for (uint32_t v = 0; v < g.n; v++) keep[v] = core[v] >= k;
    induced_subgraph(g, keep, out, old_id);
    core_out.resize(old_id.size());
    for (size_t i = 0; i < old_id.size(); i++) core_out[i] = core[old_id[i]];
}

// ----------------------------------------------------------------------------------------------------------
// output
// ----------------------------------------------------------------------------------------------------------
// One CSV row (same style as densest_row in densest.h):
//   impl,graph,n,m,triangles,vertices,density,time_ms,h2d_ms,total_ms,rounds,kmax
// n,m = size of the graph the algorithm ran on; triangles,vertices = the subgraph it returned (core rows: the kmax-core);
// density is a 6-decimal display value (decisions use the integers); total_ms = time_ms + h2d_ms; rounds = level-synchronous
// rounds (gpu core rows), passes (greedy_pp), flow calls (exact rows), ceil(lb) (prune row), else 0; kmax = maximum core number
// (core rows and prune row), else 0.
inline std::string dsg_row(const char* impl, const char* graph, uint32_t n, uint32_t m, uint64_t tri, uint64_t verts,
                           double time_ms, double h2d_ms, uint64_t rounds, uint64_t kmax) {
    char buf[1024];
    snprintf(buf, sizeof buf, "%s,%s,%u,%u,%llu,%llu,%.6f,%.3f,%.3f,%.3f,%llu,%llu", impl, graph, n, m, (unsigned long long)tri,
             (unsigned long long)verts, density_value(tri, verts), time_ms, h2d_ms, time_ms + h2d_ms, (unsigned long long)rounds,
             (unsigned long long)kmax);
    return buf;
}

// ----------------------------------------------------------------------------------------------------------
// GPU level-synchronous core decomposition: host loop (shared with the emulator test, like batch_peel_loop)
// ----------------------------------------------------------------------------------------------------------
// Round: recount tri[] on the alive vertices (tc_per_vertex_alive), m = min tri over alive (core_min_alive_tri),
// k = max(k, m), remove every alive v with tri[v] <= k and set core[v] = k (core_peel_level).
//
// WHY THIS EQUALS Alg 3. Claim: a vertex removed while the level is k has core number exactly k. Invariants (k never decreases,
// A = alive set): (a) every vertex outside A has core <= k, so the (k+1)-core is inside A; (b) every vertex of A has core >= k.
//  (b): when k is raised to m, all vertices of A have tri_A >= m = k, so A is a subgraph with minimum degree >= k, hence inside
//       the k-core; A only shrinks afterwards. At the start k = 0 and (b) is trivial.
//  (a)+(removal): v in A with tri_A(v) <= k. If core(v) >= k+1, v would be in the (k+1)-core which lies inside A, where its
//       degree is >= k+1 (min degree of the (k+1)-core) and tri_A(v) >= that - contradiction. So core(v) <= k, and with (b) it is k.
// All vertices with tri <= k are removed in the same round using the same tri[] snapshot; the argument above only uses A, so
// simultaneous removal is fine. Each round removes at least the vertex attaining the minimum (min <= k after k = max(k,min)),
// so the loop ends in at most n rounds. kmax = final k.
//
// Backend B (CUDA in gpu_dsg.cu, CPU emulator in emu_dsg.cpp):
//   B.reset_state()            alive[] = 1, core[] = UINT32_MAX
//   B.begin_round()            zero tri[] and the counters, min_tri = UINT32_MAX
//   B.run_tc()                 tc_per_vertex_alive      (stats[0] = T)
//   B.run_min()                core_min_alive_tri
//   B.run_peel(k)              core_peel_level          (stats[1] = removed, stats[2] = new k)
//   B.read_stats(T, removed, k) the ONE host sync per round
struct CoreLoopResult {
    uint32_t rounds = 0, kmax = 0;
    bool ok = true;
    std::string error;
};
template <class B>
inline CoreLoopResult core_decomp_loop(uint32_t n, uint64_t T_full, B& be, bool verbose = false, FILE* diag = nullptr) {
    CoreLoopResult r;
    if (T_full >= (1ull << 32)) { r.ok = false; r.error = "triangle count >= 2^32: uint32 tri[] counters could overflow"; return r; }
    be.reset_state();
    uint64_t nalive = n, k = 0;
    for (uint32_t round = 0; nalive > 0; round++) {
        be.begin_round();
        be.run_tc();
        be.run_min();
        be.run_peel(k);
        uint64_t T = 0, removed = 0, k_new = 0;
        be.read_stats(T, removed, k_new);
        if (round == 0 && T != T_full) { r.ok = false; r.error = "device triangle count in round 0 differs from CPU count"; return r; }
        if (verbose && diag)
            fprintf(diag, "round %6u  alive=%10llu  T=%12llu  level k=%llu  removed=%10llu\n", round, (unsigned long long)nalive,
                    (unsigned long long)T, (unsigned long long)k_new, (unsigned long long)removed);
        if (removed == 0 || removed > nalive || k_new < k) { r.ok = false; r.error = "bad stats from device (removed 0, removed > alive, or level decreased)"; return r; }
        nalive -= removed;
        k = k_new;
        r.rounds = round + 1;
    }
    r.kmax = (uint32_t)k;
    return r;
}

// Incremental version (kernels core_find_frontier / core_update_frontier / core_retire_frontier in kernels.cuh). Backend B:
//   reset_state()  all vertices alive (state 1), core = unassigned;   init_tri()  tri[] and T from ONE full triangle count;
//   begin_round()  reset the per-round counters (NOT tri[]);   run_min(); run_find(k); run_update(); run_retire();
//   read_stats(removed, k)  the single device-to-host copy per round.
// Same level rule and same stopping conditions as core_decomp_loop; the host verifies the result against Algorithm 3.
template <class B>
inline CoreLoopResult core_decomp_loop_inc(uint32_t n, uint64_t T_full, B& be, bool verbose = false, FILE* diag = nullptr) {
    CoreLoopResult r;
    if (T_full >= (1ull << 32)) { r.ok = false; r.error = "triangle count >= 2^32: uint32 tri[] counters could overflow"; return r; }
    be.reset_state();
    uint64_t T0 = 0;
    be.init_tri(T0);
    if (T0 != T_full) { r.ok = false; r.error = "device triangle count differs from CPU count"; return r; }
    uint64_t nalive = n, k = 0;
    for (uint32_t round = 0; nalive > 0; round++) {
        be.begin_round();
        be.run_min();
        be.run_find(k);
        be.run_update();
        be.run_retire();
        uint64_t removed = 0, k_new = 0;
        be.read_stats(removed, k_new);
        if (verbose && diag)
            fprintf(diag, "round %6u  alive=%10llu  level k=%llu  removed=%10llu\n", round, (unsigned long long)nalive,
                    (unsigned long long)k_new, (unsigned long long)removed);
        if (removed == 0 || removed > nalive || k_new < k) { r.ok = false; r.error = "bad stats from device (removed 0, removed > alive, or level decreased)"; return r; }
        nalive -= removed;
        k = k_new;
        r.rounds = round + 1;
    }
    r.kmax = (uint32_t)k;
    return r;
}
