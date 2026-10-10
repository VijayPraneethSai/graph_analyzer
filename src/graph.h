// graph.h - edge-list loader, degree orientation, CSR build, CPU triangle counters.
// Shared by the CPU baseline (cpu_tc.cpp) and the CUDA driver (gpu_tc.cu).
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Oriented graph in CSR form. Every undirected edge {a,b} is stored once, from the
// endpoint with the smaller (degree, id) to the other. Each triangle then has exactly
// one "lowest" vertex u with two out-neighbours v,w where w is also an out-neighbour of v,
// so triangles = sum over oriented edges (u,v) of |N+(u) ∩ N+(v)|.
struct Graph {
    uint32_t n = 0;                 // vertices (ids are 0..n-1)
    uint32_t m = 0;                 // oriented edges == undirected edges after dedup
    std::vector<uint32_t> rowptr;   // size n+1
    std::vector<uint32_t> col;      // size m, sorted within each row
    std::vector<uint32_t> src;      // size m, src[e] = row owning edge e (for edge-parallel kernels)
    uint32_t max_out_degree = 0;
};

inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Builds the oriented CSR from an UNDIRECTED edge list on vertices 0..n-1. Requirements: every pair appears
// exactly once (no duplicates, no self loops); the order of the list and the order inside a pair do not matter.
// This is the construction load_graph always used (extracted unchanged so that induced_subgraph() in dsg.h
// produces graphs that are valid in exactly the same sense): orientation from the smaller (degree, id) endpoint
// to the other, rows sorted, src[] filled, max_out_degree computed.
inline void build_graph_from_edges(uint32_t n, const std::vector<std::pair<uint32_t, uint32_t>>& edges, Graph& g) {
    g.n = n;
    std::vector<uint32_t> deg(g.n, 0);
    for (auto& e : edges) { deg[e.first]++; deg[e.second]++; }

    auto before = [&](uint32_t a, uint32_t b) {
        return deg[a] != deg[b] ? deg[a] < deg[b] : a < b;
    };
    g.m = (uint32_t)edges.size();
    g.rowptr.assign(g.n + 1, 0);
    for (auto& e : edges) {
        uint32_t from = before(e.first, e.second) ? e.first : e.second;
        g.rowptr[from + 1]++;
    }
    for (uint32_t i = 0; i < g.n; i++) g.rowptr[i + 1] += g.rowptr[i];
    g.col.resize(g.m);
    g.src.resize(g.m);
    std::vector<uint32_t> fill(g.rowptr.begin(), g.rowptr.end() - 1);
    for (auto& e : edges) {
        bool fwd = before(e.first, e.second);
        uint32_t from = fwd ? e.first : e.second, to = fwd ? e.second : e.first;
        g.col[fill[from]++] = to;
    }
    g.max_out_degree = 0;
    for (uint32_t u = 0; u < g.n; u++) {
        std::sort(g.col.begin() + g.rowptr[u], g.col.begin() + g.rowptr[u + 1]);
        for (uint32_t e = g.rowptr[u]; e < g.rowptr[u + 1]; e++) g.src[e] = u;
        g.max_out_degree = std::max(g.max_out_degree, g.rowptr[u + 1] - g.rowptr[u]);
    }
}

// Reads a SNAP-style text edge list: lines "u v", '#' starts a comment line.
// Self loops are dropped, edges are treated as undirected and de-duplicated.
inline bool load_graph(const char* path, Graph& g) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    char line[512];
    uint32_t max_id = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '%' || line[0] == '\n') continue;
        char* end;
        unsigned long a = strtoul(line, &end, 10);
        if (end == line) continue;
        char* end2;
        unsigned long b = strtoul(end, &end2, 10);
        if (end2 == end) continue;
        if (a == b) continue;
        uint32_t u = (uint32_t)std::min(a, b), v = (uint32_t)std::max(a, b);
        edges.emplace_back(u, v);
        max_id = std::max(max_id, v);
    }
    fclose(f);
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    build_graph_from_edges(max_id + 1, edges, g);
    return true;
}

// Size of the intersection of two sorted ranges.
inline uint64_t intersect_count(const uint32_t* a, uint32_t na, const uint32_t* b, uint32_t nb) {
    uint64_t c = 0;
    uint32_t i = 0, j = 0;
    while (i < na && j < nb) {
        if (a[i] < b[j]) i++;
        else if (a[i] > b[j]) j++;
        else { c++; i++; j++; }
    }
    return c;
}

inline uint64_t cpu_triangles_serial(const Graph& g) {
    uint64_t t = 0;
    for (uint32_t u = 0; u < g.n; u++) {
        const uint32_t* nu = g.col.data() + g.rowptr[u];
        uint32_t du = g.rowptr[u + 1] - g.rowptr[u];
        for (uint32_t k = 0; k < du; k++) {
            uint32_t v = nu[k];
            t += intersect_count(nu, du, g.col.data() + g.rowptr[v], g.rowptr[v + 1] - g.rowptr[v]);
        }
    }
    return t;
}

inline uint64_t cpu_triangles_omp(const Graph& g) {
    uint64_t t = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : t)
    for (long long ui = 0; ui < (long long)g.n; ui++) {
        uint32_t u = (uint32_t)ui;
        const uint32_t* nu = g.col.data() + g.rowptr[u];
        uint32_t du = g.rowptr[u + 1] - g.rowptr[u];
        for (uint32_t k = 0; k < du; k++) {
            uint32_t v = nu[k];
            t += intersect_count(nu, du, g.col.data() + g.rowptr[v], g.rowptr[v + 1] - g.rowptr[v]);
        }
    }
    return t;
}

// Splits every oriented edge's work (size of the smaller of N+(u), N+(v)) into chunks of `chunk`
// elements. Task t = (edge t_edge[t], first element t_start[t]). Used by GPU kernel v4.
inline void build_tasks(const Graph& g, uint32_t chunk, std::vector<uint32_t>& t_edge,
                        std::vector<uint32_t>& t_start) {
    t_edge.clear();
    t_start.clear();
    for (uint32_t e = 0; e < g.m; e++) {
        uint32_t u = g.src[e], v = g.col[e];
        uint32_t du = g.rowptr[u + 1] - g.rowptr[u], dv = g.rowptr[v + 1] - g.rowptr[v];
        uint32_t s = du < dv ? du : dv;
        for (uint32_t off = 0; off < s; off += chunk) { t_edge.push_back(e); t_start.push_back(off); }
    }
}

// Edge processing order for GPU kernel v9. perm[i] = edge id. Edges are counting-sorted (stable) into
// buckets of work estimate w = min(du, dv) (size of the smaller list), bucket = floor(log2(w+1)) in 0..32,
// heaviest bucket FIRST (so the longest warps start early and the light tail fills the gaps).
// O(m) time, no comparison sort.
inline void build_edge_order(const Graph& g, std::vector<uint32_t>& perm) {
    const int NB = 33;
    auto bucket = [&](uint32_t e) {
        uint32_t u = g.src[e], v = g.col[e];
        uint32_t du = g.rowptr[u + 1] - g.rowptr[u], dv = g.rowptr[v + 1] - g.rowptr[v];
        uint64_t w = (du < dv ? du : dv) + 1ull;       // >= 1
        int b = 0;
        while (w > 1) { w >>= 1; b++; }                // floor(log2(w))
        return b;
    };
    std::vector<uint32_t> start(NB + 1, 0);
    for (uint32_t e = 0; e < g.m; e++) start[(NB - 1 - bucket(e)) + 1]++;   // reversed index: heavy first
    for (int b = 0; b < NB; b++) start[b + 1] += start[b];
    perm.assign(g.m, 0);
    for (uint32_t e = 0; e < g.m; e++) perm[start[NB - 1 - bucket(e)]++] = e;
}
