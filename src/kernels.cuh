// kernels.cuh - GPU triangle counting kernels (shared by gpu_tc.cu and the CPU emulator).
//
// All kernels count  sum over oriented edges (u,v) of |N+(u) ∩ N+(v)|  on the CSR graph from graph.h.
// Neighbour lists are sorted, so membership is tested with binary search.
//
//   v1  tc_edge_per_thread : one thread per edge, two-pointer merge (simple baseline)
//   v2  tc_warp_per_edge   : one warp per edge, lanes binary-search the smaller list in the larger
//   v3  tc_block_smem      : one block per vertex u, N+(u) staged in shared memory, block reduction
//   v4  tc_task_warp       : edge work split into fixed-size tasks (host-built), one warp per task,
//                            so hub vertices no longer serialise on one warp
//   v5  tc_group_per_edge<G>: G threads (G in {1,2,4,8,16,32}, divides 32) per edge; G=32 == v2
//   v6  tc_vertex_per_thread: one thread per vertex, loops over its out-edges; needs no `src` array
//   v7  tc_edge_find_src   : like v1 but finds the edge's source vertex by binary search over rowptr
//                            instead of reading `src` (computes instead of transferring half the data)
//   v8  tc_warp_vertex_hash: one warp per vertex u; N+(u) is inserted into a per-warp open-addressing
//                            hash set in shared memory, then all lanes probe it with the elements of
//                            every N+(v), v in N+(u)  (falls back to binary search if N+(u) is too big)
//   v9  tc_edge_sorted     : like v1/v7, but edges are processed in a host-built order sorted by work
//                            estimate (min(du,dv), log2 buckets) so neighbouring threads do similar work
#pragma once
#include <cstdint>

#ifndef SM_CAP
#define SM_CAP 1024          // max |N+(u)| staged in shared memory by v3 (4 KB)
#endif
#ifndef HASH_SLOTS
#define HASH_SLOTS 512       // slots (uint32) of each warp's shared-memory hash set in v8; power of two
#endif
#ifndef V8_MAX_WARPS
#define V8_MAX_WARPS 8       // v8 static shared table holds this many warps => blockDim.x <= 32*8 = 256
#endif
#ifndef TASK_CHUNK
#define TASK_CHUNK 128       // elements of the smaller list handled per v4 task
#endif

// 1 if x is in the sorted range a[lo..hi), else 0.
__device__ __forceinline__ uint32_t contains_sorted(const uint32_t* a, uint32_t lo, uint32_t hi, uint32_t x) {
    const uint32_t end = hi;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (a[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return (lo < end && a[lo] == x) ? 1u : 0u;
}

__device__ __forceinline__ unsigned long long warp_reduce_sum(unsigned long long v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// ---- v1: one thread per oriented edge -------------------------------------------------------
__global__ void tc_edge_per_thread(const uint32_t* __restrict__ rowptr, const uint32_t* __restrict__ col,
                                   const uint32_t* __restrict__ src, uint32_t m,
                                   unsigned long long* __restrict__ total) {
    unsigned long long local = 0;
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < m; e += gridDim.x * blockDim.x) {
        uint32_t u = src[e], v = col[e];
        uint32_t i = rowptr[u], iend = rowptr[u + 1];
        uint32_t j = rowptr[v], jend = rowptr[v + 1];
        while (i < iend && j < jend) {
            uint32_t a = col[i], b = col[j];
            if (a < b) i++;
            else if (a > b) j++;
            else { local++; i++; j++; }
        }
    }
    if (local) atomicAdd(total, local);
}

// ---- v2: one warp per oriented edge ---------------------------------------------------------
// Lanes stride over the smaller of N+(u), N+(v) and binary-search each element in the larger one.
// Counts stay in registers; one warp reduction and one atomic per warp at the end.
__global__ void tc_warp_per_edge(const uint32_t* __restrict__ rowptr, const uint32_t* __restrict__ col,
                                 const uint32_t* __restrict__ src, uint32_t m,
                                 unsigned long long* __restrict__ total) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t nwarps = (gridDim.x * blockDim.x) >> 5;
    unsigned long long local = 0;
    for (uint32_t e = gwarp; e < m; e += nwarps) {
        uint32_t u = src[e], v = col[e];
        uint32_t ru = rowptr[u], du = rowptr[u + 1] - ru;
        uint32_t rv = rowptr[v], dv = rowptr[v + 1] - rv;
        uint32_t aS, aN, bS, bN;                       // a = smaller list, b = larger
        if (du <= dv) { aS = ru; aN = du; bS = rv; bN = dv; }
        else          { aS = rv; aN = dv; bS = ru; bN = du; }
        for (uint32_t i = lane; i < aN; i += 32)
            local += contains_sorted(col, bS, bS + bN, col[aS + i]);
    }
    local = warp_reduce_sum(local);
    if (lane == 0 && local) atomicAdd(total, local);
}

// ---- v3: one block per vertex u, N+(u) cached in shared memory -------------------------------
// N+(u) is reused by every out-edge (u,v), so it is loaded from global memory once per block.
// If |N+(u)| > SM_CAP the search falls back to global memory. blockDim.x must be a multiple of 32, <= 1024.
__global__ void tc_block_smem(const uint32_t* __restrict__ rowptr, const uint32_t* __restrict__ col,
                              uint32_t n, unsigned long long* __restrict__ total) {
    __shared__ uint32_t s_list[SM_CAP];
    __shared__ unsigned long long s_warp[32];
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = blockDim.x >> 5;
    unsigned long long local = 0;
    for (uint32_t u = blockIdx.x; u < n; u += gridDim.x) {
        uint32_t ru = rowptr[u], du = rowptr[u + 1] - ru;
        if (du == 0) continue;                        // same for every thread of the block
        const bool cached = du <= SM_CAP;
        if (cached)
            for (uint32_t i = threadIdx.x; i < du; i += blockDim.x) s_list[i] = col[ru + i];
        __syncthreads();
        for (uint32_t k = warp; k < du; k += nwarps) {
            uint32_t v = col[ru + k];
            uint32_t rv = rowptr[v], dv = rowptr[v + 1] - rv;
            for (uint32_t i = lane; i < dv; i += 32) {
                uint32_t x = col[rv + i];
                local += cached ? contains_sorted(s_list, 0, du, x)
                                : contains_sorted(col, ru, ru + du, x);
            }
        }
        __syncthreads();                              // s_list is overwritten by the next u
    }
    local = warp_reduce_sum(local);
    if (lane == 0) s_warp[warp] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long sum = 0;
        for (uint32_t w = 0; w < nwarps; w++) sum += s_warp[w];
        if (sum) atomicAdd(total, sum);
    }
}

// ---- v4: fixed-size tasks, one warp per task ------------------------------------------------
// Host splits every edge's work (size of the smaller list) into chunks of TASK_CHUNK elements:
// task t = (edge t_edge[t], first element t_start[t]). Hub-heavy edges become many tasks that are
// spread over all warps instead of keeping one warp busy.
__global__ void tc_task_warp(const uint32_t* __restrict__ rowptr, const uint32_t* __restrict__ col,
                             const uint32_t* __restrict__ src,
                             const uint32_t* __restrict__ t_edge, const uint32_t* __restrict__ t_start,
                             uint32_t ntasks, unsigned long long* __restrict__ total) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t nwarps = (gridDim.x * blockDim.x) >> 5;
    unsigned long long local = 0;
    for (uint32_t t = gwarp; t < ntasks; t += nwarps) {
        uint32_t e = t_edge[t], first = t_start[t];
        uint32_t u = src[e], v = col[e];
        uint32_t ru = rowptr[u], du = rowptr[u + 1] - ru;
        uint32_t rv = rowptr[v], dv = rowptr[v + 1] - rv;
        uint32_t aS, aN, bS, bN;
        if (du <= dv) { aS = ru; aN = du; bS = rv; bN = dv; }
        else          { aS = rv; aN = dv; bS = ru; bN = du; }
        uint32_t last = first + TASK_CHUNK;
        if (last > aN) last = aN;
        for (uint32_t i = first + lane; i < last; i += 32)
            local += contains_sorted(col, bS, bS + bN, col[aS + i]);
    }
    local = warp_reduce_sum(local);
    if (lane == 0 && local) atomicAdd(total, local);
}

// ---- v5: G threads per oriented edge (G divides 32) -----------------------------------------
// After degree orientation out-lists are short (max out-degree ~100-200, typical a handful), so in v2
// most of the 32 lanes of a warp idle (smaller list < 32 elements) and the warp is mostly wasted.
// With a group of G threads, 32/G edges are processed per warp at once: each thread strides the
// smaller list by G starting at its sub-lane, binary-searching in the larger list. Smaller G means
// less idle lanes on short lists, at the cost of more serial iterations on long ones, and the
// edges of one warp may differ in length (some divergence). A group never spans warps since G | 32.
// Counts stay in per-thread registers; the warp reduction is done once after the loop by all 32 lanes
// (it must not sit inside the divergent edge loop because __shfl_down_sync uses the full mask).
template <int G>
__global__ void tc_group_per_edge(const uint32_t* __restrict__ rowptr, const uint32_t* __restrict__ col,
                                  const uint32_t* __restrict__ src, uint32_t m,
                                  unsigned long long* __restrict__ total) {
    static_assert(G >= 1 && G <= 32 && (32 % G) == 0, "G must divide 32");
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t group = tid / G;
    const uint32_t sub = threadIdx.x % G;
    const uint32_t ngroups = (gridDim.x * blockDim.x) / G;
    const uint32_t lane = threadIdx.x & 31;
    unsigned long long local = 0;
    for (uint32_t e = group; e < m; e += ngroups) {
        uint32_t u = src[e], v = col[e];
        uint32_t ru = rowptr[u], du = rowptr[u + 1] - ru;
        uint32_t rv = rowptr[v], dv = rowptr[v + 1] - rv;
        uint32_t aS, aN, bS, bN;                       // a = smaller list, b = larger
        if (du <= dv) { aS = ru; aN = du; bS = rv; bN = dv; }
        else          { aS = rv; aN = dv; bS = ru; bN = du; }
        for (uint32_t i = sub; i < aN; i += G)
            local += contains_sorted(col, bS, bS + bN, col[aS + i]);
    }
    local = warp_reduce_sum(local);
    if (lane == 0 && local) atomicAdd(total, local);
}

// ---- v6: one thread per vertex u (no src array needed) ---------------------------------------
// Thread u walks its own out-edges col[rowptr[u]..rowptr[u+1]) and merges N+(u) with N+(v) exactly like
// v1. The source vertex is simply the loop variable, so the `src` array (m uint32, half of the H2D
// traffic of v1) is never uploaded. Trade-off: work per thread = out-degree * list length, so load is
// unbalanced across the warp (divergence) and hub vertices serialise on one thread.
__global__ void tc_vertex_per_thread(const uint32_t* __restrict__ rowptr, const uint32_t* __restrict__ col,
                                     uint32_t n, unsigned long long* __restrict__ total) {
    unsigned long long local = 0;
    for (uint32_t u = blockIdx.x * blockDim.x + threadIdx.x; u < n; u += gridDim.x * blockDim.x) {
        const uint32_t ubeg = rowptr[u], uend = rowptr[u + 1];
        for (uint32_t e = ubeg; e < uend; e++) {
            uint32_t v = col[e];
            uint32_t i = ubeg, j = rowptr[v], jend = rowptr[v + 1];
            while (i < uend && j < jend) {
                uint32_t a = col[i], b = col[j];
                if (a < b) i++;
                else if (a > b) j++;
                else { local++; i++; j++; }
            }
        }
    }
    if (local) atomicAdd(total, local);
}

// ---- v7: one thread per edge, source vertex found by binary search (no src array) -------------
// Compute-for-transfer trade: instead of uploading src[e] (m uint32, half of v1's H2D bytes), each
// thread recovers u = the vertex with rowptr[u] <= e < rowptr[u+1] by binary search over rowptr
// (log2(n) ~ 20 dependent loads, mostly L2 hits since neighbouring threads search nearby edges).
// Vertices with no out-edges have rowptr[u] == rowptr[u+1]; the upper-bound search (first index whose
// rowptr is > e, minus one) skips them and returns the last u with rowptr[u] <= e, which then
// satisfies e < rowptr[u+1] because rowptr[n] == m > e. The merge is the same as v1.
__global__ void tc_edge_find_src(const uint32_t* __restrict__ rowptr, const uint32_t* __restrict__ col,
                                 uint32_t n, uint32_t m, unsigned long long* __restrict__ total) {
    unsigned long long local = 0;
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < m; e += gridDim.x * blockDim.x) {
        uint32_t lo = 0, hi = n;                       // invariant: answer idx in [lo, hi]
        while (lo < hi) {                              // idx = first index in [0,n] with rowptr[idx] > e
            uint32_t mid = lo + ((hi - lo) >> 1);
            if (rowptr[mid] <= e) lo = mid + 1;
            else hi = mid;
        }
        uint32_t u = lo - 1;                           // lo >= 1 because rowptr[0] == 0 <= e
        uint32_t v = col[e];
        uint32_t i = rowptr[u], iend = rowptr[u + 1];
        uint32_t j = rowptr[v], jend = rowptr[v + 1];
        while (i < iend && j < jend) {
            uint32_t a = col[i], b = col[j];
            if (a < b) i++;
            else if (a > b) j++;
            else { local++; i++; j++; }
        }
    }
    if (local) atomicAdd(total, local);
}

// ---- v8: one warp per vertex u, N+(u) in a per-warp shared-memory hash set ---------------------
// Why this should help (profiling of v1/v2 on the T4: after degree orientation lists are short, v1 has
// only ~11/32 lanes active and is L1/TEX bound; v2 executes ~2.6x more instructions because 32 lanes per
// short list is wasteful):
//   * the 32 lanes work on the neighbour lists of ALL v in N+(u) together: lane i takes elements i, i+32,
//     ... of each N+(v) and the lanes' loads are consecutive col[] addresses (coalesced). Unlike v2 the
//     warp is not tied to ONE edge; but note a list shorter than 32 still leaves lanes idle for that
//     iteration (lists are visited one after another), so the gain is mainly the cheaper test below.
//   * N+(u) is read from global memory ONCE per vertex (to build the table) instead of once per out-edge.
//   * a membership test is ~1-2 shared-memory probes instead of a ~log2(du) chain of dependent global loads.
// What can go wrong:
//   * shared-memory bank conflicts: lanes probe random slots, two lanes in the same bank (slot % 32)
//     serialise; expected ~ a few-way conflicts on average, not 32-way.
//   * atomicCAS on shared memory is slower than a plain store; insertion cost is paid once per (u, element).
//   * clearing HASH_SLOTS slots costs HASH_SLOTS/32 stores per lane for EVERY vertex, even one with
//     out-degree 1, so low-degree vertices pay a fixed overhead. Vertices with out-degree 0 skip everything
//     (the test is warp-uniform, so no divergence and no __syncwarp deadlock).
//   * smem per block = V8_MAX_WARPS*HASH_SLOTS*4 B (16 KB by default, STATIC); this limits occupancy.
// Table layout: open addressing, linear probing, HASH_SLOTS power of two, empty = 0xFFFFFFFF (vertex ids
// are < n < 2^32-1). Load factor is kept <= 0.5 by the cap du <= HASH_SLOTS/2, which also guarantees
// every probe sequence reaches an empty slot, i.e. lookups terminate.
// If du > HASH_SLOTS/2 the warp uses the fallback: binary-search each element of N+(v) in N+(u) in global
// memory (like v2), so correctness never depends on the cap. blockDim.x must be a multiple of 32 and
// <= 32*V8_MAX_WARPS.
#define V8_EMPTY 0xFFFFFFFFu
__device__ __forceinline__ uint32_t hash_slot(uint32_t x) {
    uint32_t h = x * 0x9E3779B1u;                      // Fibonacci/multiplicative hash ...
    h ^= h >> 15;                                      // ... then fold high bits down, since only the low bits are used
    return h & (HASH_SLOTS - 1);
}
__global__ void tc_warp_vertex_hash(const uint32_t* __restrict__ rowptr, const uint32_t* __restrict__ col,
                                    uint32_t n, unsigned long long* __restrict__ total) {
    static_assert((HASH_SLOTS & (HASH_SLOTS - 1)) == 0 && HASH_SLOTS >= 2, "HASH_SLOTS must be a power of two");
    __shared__ uint32_t s_tab[V8_MAX_WARPS * HASH_SLOTS];
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t wib = threadIdx.x >> 5;             // warp index inside the block
    uint32_t* tab = s_tab + wib * HASH_SLOTS;          // this warp's private table
    const uint32_t gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t nwarps = (gridDim.x * blockDim.x) >> 5;
    unsigned long long local = 0;
    for (uint32_t u = gwarp; u < n; u += nwarps) {
        const uint32_t ru = rowptr[u], du = rowptr[u + 1] - ru;
        if (du == 0) continue;                         // warp-uniform: cheap skip, no clear / no sync
        if (du <= HASH_SLOTS / 2) {
            for (uint32_t s = lane; s < HASH_SLOTS; s += 32) tab[s] = V8_EMPTY;   // consecutive banks: no conflict
            __syncwarp();                              // table must be fully cleared before inserts
            for (uint32_t i = lane; i < du; i += 32) {
                const uint32_t x = col[ru + i];        // coalesced read of N+(u)
                uint32_t h = hash_slot(x);
                while (true) {
                    uint32_t old = atomicCAS(&tab[h], V8_EMPTY, x);
                    if (old == V8_EMPTY || old == x) break;   // inserted (lists have no duplicates)
                    h = (h + 1) & (HASH_SLOTS - 1);    // linear probing
                }
            }
            __syncwarp();                              // all inserts visible before any lookup
            for (uint32_t k = 0; k < du; k++) {        // warp-uniform loop over out-neighbours v
                const uint32_t v = col[ru + k];        // same address for all lanes: broadcast
                const uint32_t rv = rowptr[v], dv = rowptr[v + 1] - rv;
                for (uint32_t i = lane; i < dv; i += 32) {
                    const uint32_t x = col[rv + i];    // coalesced; x in N+(v)
                    uint32_t h = hash_slot(x);
                    while (true) {                     // probe until hit or empty slot
                        const uint32_t t = tab[h];
                        if (t == x) { local++; break; }
                        if (t == V8_EMPTY) break;
                        h = (h + 1) & (HASH_SLOTS - 1);
                    }
                }
            }
            __syncwarp();                              // lookups done before the next u clears the table
        } else {                                       // fallback: N+(u) too big for the table
            for (uint32_t k = 0; k < du; k++) {
                const uint32_t v = col[ru + k];
                const uint32_t rv = rowptr[v], dv = rowptr[v + 1] - rv;
                for (uint32_t i = lane; i < dv; i += 32)
                    local += contains_sorted(col, ru, ru + du, col[rv + i]);
            }
        }
    }
    local = warp_reduce_sum(local);
    if (lane == 0 && local) atomicAdd(total, local);
}

// ---- v9: one thread per edge, edges visited in a work-sorted order -----------------------------
// Same merge as v1, but thread i handles edge perm[i]. perm (host-built, build_edge_order in graph.h) is
// a counting sort of the edges by floor(log2(min(du,dv)+1)), so the 32 threads of a warp get edges with
// similar merge length: in v1 a warp runs as long as its slowest lane and the others sit idle (divergence).
// Cost: an extra indirection perm[i] (coalesced) and then scattered src/col/rowptr reads, because
// neighbouring threads no longer handle neighbouring edges (worse locality), plus m*4 bytes of H2D and the
// host sort time. Whether the divergence gain beats that is exactly what must be measured.
__global__ void tc_edge_sorted(const uint32_t* __restrict__ rowptr, const uint32_t* __restrict__ col,
                               const uint32_t* __restrict__ src, const uint32_t* __restrict__ perm,
                               uint32_t m, unsigned long long* __restrict__ total) {
    unsigned long long local = 0;
    for (uint32_t i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < m; i0 += gridDim.x * blockDim.x) {
        const uint32_t e = perm[i0];
        uint32_t u = src[e], v = col[e];
        uint32_t i = rowptr[u], iend = rowptr[u + 1];
        uint32_t j = rowptr[v], jend = rowptr[v + 1];
        while (i < iend && j < jend) {
            uint32_t a = col[i], b = col[j];
            if (a < b) i++;
            else if (a > b) j++;
            else { local++; i++; j++; }
        }
    }
    if (local) atomicAdd(total, local);
}

// =====================================================================================================
// Triangle-density densest subgraph: kernels for GPU batch peeling (see src/densest.h for the algorithm
// and docs in README "GPU densest subgraph pipeline").
//
// One peeling round = (1) reset tri[] and the two counters, (2) tc_per_vertex_alive, (3) mark_peel.
// The vertex set is never compacted: `alive[v]` (1 byte per vertex) says which vertices still count, and
// the CSR arrays (rowptr/col/src) stay exactly as uploaded. That is the simplest correct design; its price
// is that late rounds (few alive vertices) still launch one thread per ORIGINAL edge, and most of those
// threads exit right after reading alive[u] and alive[v]. Compacting the edge list every few rounds is the
// obvious next optimisation (not done, so we do not claim anything about it).
// =====================================================================================================

// ---- per-vertex triangle participation in the subgraph induced by the alive vertices -----------------
// Same structure as v1 (one thread per oriented edge (u,v), two-pointer merge of N+(u) and N+(v)), plus:
//   * an edge is skipped if u or v is dead (2 byte loads; dead edges cost almost nothing);
//   * a common neighbour w only counts as a triangle if w is alive too.
// Every triangle {a,b,c} of the induced subgraph is found exactly once (at its lowest-ranked vertex's edge to
// the middle vertex, as in v1), and then all THREE of its corners get +1:
//     tri[u]++, tri[v]++, tri[w]++      =>  sum_v tri[v] = 3 * T   (T = total triangles of the induced subgraph)
// The total T is accumulated in a register and flushed with ONE atomicAdd per thread (like v1).
//
// Memory/warp behaviour (reasoning, not a measurement):
//   * col[i], col[j] merge reads: each lane walks its own two lists, so lanes of a warp touch unrelated
//     addresses (uncoalesced) and finish after different numbers of steps (divergence) - same as v1.
//   * alive[w] is a 1-byte load at an arbitrary address per triangle (uncoalesced, but the whole alive[] array
//     is only n bytes, so it mostly stays in L2).
//   * tri[] atomics: 3 per triangle, to random addresses. Contention trade-off: a hub vertex that lies in
//     millions of triangles receives millions of atomicAdds on ONE address, which the L2 atomic unit
//     serialises. Atomics on distinct addresses run in parallel, so low-participation vertices are cheap.
//     We accept this: it is the simplest exact method and uint32 atomics are native (no CAS loop).
//
//   ALTERNATIVE IDEA (comments only, NOT implemented here): u and v are the same for every triangle found in
//   one edge's merge, so count `c` locally and do just two atomics per EDGE (tri[u]+=c, tri[v]+=c) plus one per
//   triangle for w. That removes about 2/3 of the atomics when edges carry many triangles. Another option is a
//   warp-aggregated atomic (lanes that hit the same vertex combine via __match_any_sync before one atomicAdd).
//   Neither is needed for correctness, and neither is measured; add them only as separate, benchmarked stages.
//
// Bounds: tri[v] <= T <= 2^32-1 is required (uint32 counters); densest.h's peel_arith_ok() checks it on the host.
// `tri` must be zeroed (cudaMemset) before every launch; `total` must be zeroed too.
__global__ void tc_per_vertex_alive(const uint32_t* __restrict__ rowptr, const uint32_t* __restrict__ col,
                                    const uint32_t* __restrict__ src, const uint8_t* __restrict__ alive,
                                    uint32_t m, uint32_t* tri, unsigned long long* __restrict__ total) {
    unsigned long long local = 0;
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < m; e += gridDim.x * blockDim.x) {
        const uint32_t u = src[e], v = col[e];
        if (!alive[u] || !alive[v]) continue;          // edge not in the induced subgraph
        uint32_t i = rowptr[u], iend = rowptr[u + 1];
        uint32_t j = rowptr[v], jend = rowptr[v + 1];
        while (i < iend && j < jend) {
            const uint32_t a = col[i], b = col[j];
            if (a < b) i++;
            else if (a > b) j++;
            else {                                      // a == b: common out-neighbour w closes triangle (u,v,w)
                const uint32_t w = a;
                if (alive[w]) {
                    local++;
                    atomicAdd(&tri[u], 1u);
                    atomicAdd(&tri[v], 1u);
                    atomicAdd(&tri[w], 1u);
                }
                i++; j++;
            }
        }
    }
    if (local) atomicAdd(total, local);
}

// ---- peel step: remove every alive vertex whose participation is at most (1+eps) * average ----------
// With T = *total (triangles of the current alive subgraph, written by tc_per_vertex_alive in the same
// stream, so it is complete when this kernel starts) and nalive alive vertices, average participation is
// 3T/nalive. eps = num/den, so the removal test  tri[v] <= (1+eps) * 3T/nalive  becomes, WITHOUT division or
// floating point:
//        tri[v] * nalive * den  <=  3 * (den + num) * T                      (all uint64)
// Overflow bound: tri[v] <= T0 < 2^32 (T0 = triangles of the full graph), nalive <= n < 2^32 and den <= 2^20 give
// a left side < 2^84 in the worst case, so the host checks the REAL numbers with peel_arith_ok() (e.g. tri 1e8,
// nalive 4e6, den 10 gives 4e15, far below 1.8e19). The right side is 3*(den+num)*T0, also checked.
//
// Every thread decides from the SAME inputs (tri[], nalive, T): removing a vertex here does not change tri[]
// or nalive, which are only updated by the next round's tc_per_vertex_alive. So the result does not depend on
// thread order, and each alive[v]/removed_round[v] is written by exactly one thread (no race).
// Grid-stride loop over vertices (consecutive threads read consecutive tri[v], alive[v]: coalesced). The count
// of removed vertices is reduced in registers -> warp shuffle -> ONE atomicAdd per warp. Every thread must
// reach warp_reduce_sum (full-mask shuffles), so there is no early return before it.
__global__ void mark_peel(const uint32_t* __restrict__ tri, uint8_t* __restrict__ alive,
                          uint32_t* __restrict__ removed_round, uint32_t n,
                          unsigned long long nalive, unsigned long long num, unsigned long long den,
                          uint32_t round, const unsigned long long* __restrict__ total,
                          unsigned long long* __restrict__ removed_count) {
    const unsigned long long rhs = 3ull * (den + num) * (*total);   // same for all threads (broadcast load)
    unsigned long long local = 0;
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += gridDim.x * blockDim.x) {
        if (alive[v] && (unsigned long long)tri[v] * nalive * den <= rhs) {
            alive[v] = 0;
            removed_round[v] = round;
            local++;
        }
    }
    local = warp_reduce_sum(local);
    if ((threadIdx.x & 31) == 0 && local) atomicAdd(removed_count, local);
}

// =====================================================================================================
// Triangle (k,Psi)-core decomposition, level-synchronous (see core_decomp_loop in src/dsg.h for the loop and the
// proof that it reproduces the core numbers of Algorithm 3 of Fang et al.).
//
// One round = reset tri[]/counters, tc_per_vertex_alive (above, reused unchanged), core_min_alive_tri, core_peel_level.
// The minimum has to be known by ALL threads before anybody decides, and CUDA has no grid-wide barrier inside a kernel
// launch, so "min over alive" and "peel" are two kernels (the kernel boundary is the barrier), in the same stream so no
// host synchronisation sits between them.
// =====================================================================================================

// Warp minimum with the same shuffle pattern as warp_reduce_sum: every lane must call it (full-mask shuffles).
__device__ __forceinline__ unsigned warp_reduce_min(unsigned v) {
    for (int off = 16; off > 0; off >>= 1) {
        const unsigned o = __shfl_down_sync(0xffffffffu, v, off);
        v = o < v ? o : v;
    }
    return v;                                           // lane 0 holds the minimum of the 32 lanes
}

// *min_tri = min over alive v of tri[v]. *min_tri must be initialised to 0xFFFFFFFF (cudaMemset 0xFF).
// Grid-stride loop: consecutive threads read consecutive tri[v] / alive[v] (coalesced). Each thread keeps a register minimum,
// the warp reduces with shuffles (no shared memory, no barrier), and lane 0 issues ONE atomicMin per warp - so the number of
// atomics is n/32 at most instead of n, and they all hit one address, which is fine for a once-per-round kernel.
__global__ void core_min_alive_tri(const uint32_t* __restrict__ tri, const uint8_t* __restrict__ alive, uint32_t n,
                                   unsigned* min_tri) {
    unsigned local = 0xFFFFFFFFu;
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += gridDim.x * blockDim.x)
        if (alive[v] && tri[v] < local) local = tri[v];
    local = warp_reduce_min(local);                     // all lanes participate: no early return above
    if ((threadIdx.x & 31) == 0 && local != 0xFFFFFFFFu) atomicMin(min_tri, local);
}

// Level step: k = max(k_in, *min_tri); every alive v with tri[v] <= k is removed and gets core[v] = k.
// All threads compute the same k from the same inputs (k_in is a kernel argument, *min_tri is only READ here), so the
// decision for a vertex does not depend on thread order, and each alive[v]/core[v] has exactly one writer (the thread that
// owns v in the grid-stride loop). tri[] is not modified, so a vertex removed by one thread does not change the decision of
// another: this is the "simultaneous removal from one snapshot" the correctness argument needs.
// removed count: register -> warp shuffle -> one atomicAdd per warp (as in mark_peel). Thread 0 of block 0 publishes the new
// level k so the host needs only ONE device-to-host copy per round (no second sync to learn k).
// tri[v] is uint32 (T < 2^32 is checked on the host); the comparison is done in 64 bits.
__global__ void core_peel_level(const uint32_t* __restrict__ tri, uint8_t* __restrict__ alive, uint32_t* __restrict__ core,
                                uint32_t n, unsigned long long k_in, const unsigned* __restrict__ min_tri,
                                unsigned long long* removed_count, unsigned long long* k_out) {
    const unsigned long long m = *min_tri;              // broadcast load, same address for all threads
    const unsigned long long k = m > k_in ? m : k_in;
    unsigned long long local = 0;
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += gridDim.x * blockDim.x) {
        if (alive[v] && (unsigned long long)tri[v] <= k) {
            alive[v] = 0;
            core[v] = (uint32_t)k;
            local++;
        }
    }
    local = warp_reduce_sum(local);
    if ((threadIdx.x & 31) == 0 && local) atomicAdd(removed_count, local);
    if (blockIdx.x == 0 && threadIdx.x == 0) *k_out = k;
}

// =====================================================================================================
// Triangle (k,Psi)-core decomposition, INCREMENTAL / frontier version (stage 6c).
//
// Why: the recount version above re-runs tc_per_vertex_alive (all triangles) on every round. The first T4 run showed what
// that costs: as-Skitter needs 5,626 rounds and the GPU took 27.2 s against 13.3 s for the serial CPU Algorithm 3 (see
// README). The work per round has to be proportional to what the round REMOVES, not to the whole graph.
//
// Idea (same as the CPU Algorithm 3, but a whole level at a time): keep tri[v] = number of triangles of v inside the alive
// subgraph. When a set F of vertices (the "frontier": every alive vertex with tri <= k) is removed, only triangles that
// contain a vertex of F disappear, so only tri[] of the alive corners of those triangles changes. We enumerate exactly those
// triangles once, from the frontier vertices' adjacency lists, and subtract.
//
// State per vertex (one byte): 0 = removed in an earlier round, 1 = alive, 2 = in this round's frontier.
//   round = core_min_alive_tri (min over state==1)  ->  core_find_frontier  ->  core_update_frontier  ->  core_retire_frontier.
// Kernel boundaries are the barriers: find reads tri[] and writes state/core/frontier; update reads state and writes tri[]
// (only atomically, only for state==1 vertices); retire turns state 2 into 0 afterwards.
//
// A triangle {a,b,c} that contains frontier vertices F' (1..3 of them) must reduce tri[] of its NON-frontier corners by 1,
// exactly once. Several frontier vertices find the same triangle (each from its own adjacency list), so the rule is: the
// triangle is handled only by its SMALLEST-id frontier member. Exactly one thread satisfies that, so no double subtraction.
// Frontier corners are not updated: they are removed this round with core = k, their tri[] is never read again.
// This reproduces the recount loop exactly (tri[x] after the round = triangles of x inside the alive subgraph after removal),
// so the core numbers are identical to Algorithm 3; the host verifies that element by element.
// =====================================================================================================

// Frontier = every alive vertex with tri[v] <= k, k = max(k_in, min over alive of tri) (same level rule as core_peel_level).
// Each such vertex gets state 2, core[v] = k and is appended to frontier[] (position from ONE atomicAdd on *fcount).
// Every vertex joins the frontier exactly once over the whole run, so the total number of these atomics is n; a warp-aggregated
// append (ballot + one atomic per warp) would save little and needs warp-vote intrinsics, so it is not used.
// Each state[v]/core[v] has one writer (the grid-stride thread that owns v); tri[] and *min_tri are read-only here.
__global__ void core_find_frontier(const uint32_t* __restrict__ tri, uint8_t* __restrict__ state, uint32_t* __restrict__ core,
                                   uint32_t n, unsigned long long k_in, const unsigned* __restrict__ min_tri,
                                   uint32_t* __restrict__ frontier, unsigned long long* fcount, unsigned long long* k_out) {
    const unsigned long long mt = *min_tri;
    const unsigned long long k = mt > k_in ? mt : k_in;
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += gridDim.x * blockDim.x) {
        if (state[v] == 1 && (unsigned long long)tri[v] <= k) {
            state[v] = 2;
            core[v] = (uint32_t)k;
            frontier[atomicAdd(fcount, 1ull)] = v;     // order inside the frontier is arbitrary; the result does not depend on it
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) *k_out = k;
}

// First index i in the sorted range a[0..len) with a[i] > x (binary search).
__device__ __forceinline__ uint32_t upper_bound_u32(const uint32_t* __restrict__ a, uint32_t len, uint32_t x) {
    uint32_t lo = 0, hi = len;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (a[mid] <= x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// One WARP per frontier vertex u (grid-stride over frontier entries). Uses the UNDIRECTED sorted adjacency (urow/ucol).
// The lanes of the warp split u's neighbour list: lane l takes neighbours at positions l, l+32, ... For a neighbour v
// (skipped if removed earlier) the triangles {u,v,w} with w > v are the common elements of N(u)[after v] and N(v)[after v], found by
// a two-pointer merge (both sorted). Each triangle at u is listed once because we require v < w.
// Reasoning: the warp-per-vertex split keeps the 32 lanes busy on one vertex's list (u's list is read coalesced,
// ucol[p] for consecutive lanes), but the merges against N(v) are per-lane and uncoalesced and lanes finish at different
// times (divergence) - the same trade-off as kernels v1-v8. A frontier vertex with a huge degree is handled by one warp only,
// which is a load-balance limit (hubs are removed late and rarely many at once); not measured, so no claim.
// Atomics: atomicSub on tri[] of non-frontier corners only (state 1); contention only on hubs.
__global__ void core_update_frontier(const uint32_t* __restrict__ urow, const uint32_t* __restrict__ ucol,
                                     const uint8_t* __restrict__ state, const uint32_t* __restrict__ frontier,
                                     const unsigned long long* __restrict__ fcount, uint32_t* tri) {
    const unsigned long long F = *fcount;
    const uint32_t lane = threadIdx.x & 31;
    const uint64_t warp0 = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint64_t nwarps = ((uint64_t)gridDim.x * blockDim.x) >> 5;
    for (uint64_t i = warp0; i < F; i += nwarps) {
        const uint32_t u = frontier[i];
        const uint32_t ub = urow[u], ue = urow[u + 1];
        for (uint32_t p = ub + lane; p < ue; p += 32) {
            const uint32_t v = ucol[p];
            const uint8_t sv = state[v];
            if (sv == 0) continue;                                   // v already removed: no triangle through v exists any more
            const uint32_t vb = urow[v], ve = urow[v + 1];
            uint32_t i1 = p + 1;                                     // N(u) entries after v are all > v (sorted, no duplicates)
            uint32_t j1 = vb + upper_bound_u32(ucol + vb, ve - vb, v);
            while (i1 < ue && j1 < ve) {
                const uint32_t a = ucol[i1], b = ucol[j1];
                if (a < b) i1++;
                else if (a > b) j1++;
                else {                                               // w = a is adjacent to both u and v, and w > v
                    const uint8_t sw = state[a];
                    if (sw != 0) {                                   // triangle {u,v,a} lies inside (alive + frontier) vertices
                        const bool smaller_frontier_member = (sv == 2 && v < u) || (sw == 2 && a < u);
                        if (!smaller_frontier_member) {              // u is the smallest-id frontier vertex of this triangle
                            if (sv == 1) atomicSub(&tri[v], 1u);
                            if (sw == 1) atomicSub(&tri[a], 1u);
                        }
                    }
                    i1++; j1++;
                }
            }
        }
    }
}

// state 2 -> 0 for this round's frontier (after core_update_frontier has finished reading state[]).
__global__ void core_retire_frontier(uint8_t* __restrict__ state, const uint32_t* __restrict__ frontier,
                                     const unsigned long long* __restrict__ fcount) {
    const unsigned long long F = *fcount;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < F; i += (unsigned long long)gridDim.x * blockDim.x)
        state[frontier[i]] = 0;
}
