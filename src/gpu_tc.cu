// gpu_tc.cu - GPU triangle counting driver: runs kernels v1..v9 from kernels.cuh.
// usage: gpu_tc <edge_list.txt> [repeats] [--pinned] [--no-burnin]   (flags may appear anywhere)
// Before timing, the GPU is kept busy ~1.5 s (burn-in) so clocks are ramped; --no-burnin disables that.
// Variants: v1, v2, v3, v4, and v5 (tc_group_per_edge) with G = 2, 4, 8, 16 threads per edge
//   (names gpu_v5_g<G>_per_edge; G = 32 is equivalent to v2 and is not run separately),
//   v6 (gpu_v6_vertex_per_thread) and v7 (gpu_v7_edge_search_src); v6/v7 never read `src`,
//   v8 (gpu_v8_warp_vertex_hash: one warp per vertex, per-warp shared-memory hash set of N+(u); reads
//   rowptr+col only; STATIC shared memory V8_MAX_WARPS*HASH_SLOTS*4 B = 16 KB per block by default, so the
//   block size is fixed at 32*V8_MAX_WARPS = 256 threads) and v9 (gpu_v9_edge_sorted: v1 with edges
//   visited in a host-built work-sorted order `perm`; reads rowptr+col+src+perm).
// --pinned: each host array is first copied (NOT timed) into page-locked memory (cudaMallocHost) and the
//   timed cudaMemcpy H2D reads from that buffer; every printed impl name then gets a "_pinned" suffix so
//   pinned and pageable rows never mix in results.csv.
// Prints one CSV line per kernel:
//   impl,graph,n,m,triangles,kernel_ms,h2d_ms,gpu_total_ms
// kernel_ms   = best-of-N, warm (one untimed warm-up launch first), launch overhead included.
// h2d_ms      = host-to-device copies of exactly the arrays that variant reads (v3 does not need `src`;
//               v4 also uploads its task list; v5 reads rowptr, col, src like v1/v2;
//               v6, v7 and v8 need only rowptr + col; v9 also uploads perm).
// gpu_total_ms= kernel_ms + h2d_ms (+ host time to build the task list for v4 / the edge order for v9). File loading and
//               graph orientation are common to all variants and are NOT included (printed on stderr).
// Each count is checked against the serial CPU count; exit code 1 if any kernel mismatches.
#include <cuda_runtime.h>
#include <functional>
#include <string>
#include "graph.h"
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

static bool g_no_burnin = false; // --no-burnin: skip the 1.5 s clock-ramp burn-in (old behaviour)
static bool g_pinned = false;   // --pinned: stage host data in page-locked memory before the timed copy

template <class T>
static T* upload(const std::vector<T>& v, float* h2d_ms_accum, cudaEvent_t e0, cudaEvent_t e1) {
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
    *h2d_ms_accum += ms;
    if (pin) CUDA_CHECK(cudaFreeHost(pin));
    return d;
}

template <int G>
static std::function<void()> make_v5_launch(int grid, int block, const uint32_t* rp, const uint32_t* cl,
                                            const uint32_t* sr, uint32_t m, unsigned long long* total) {
    return [=] { tc_group_per_edge<G><<<grid, block>>>(rp, cl, sr, m, total); };
}

int main(int argc, char** argv) {
    std::vector<const char*> pos;                      // positional args: graph path, optional repeats
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--pinned") == 0) g_pinned = true;
        else if (strcmp(argv[i], "--no-burnin") == 0) g_no_burnin = true;
        else pos.push_back(argv[i]);
    }
    if (pos.empty()) { fprintf(stderr, "usage: %s graph.txt [repeats] [--pinned]\n", argv[0]); return 2; }
    const char* graph_path = pos[0];
    const int repeats = std::max(1, pos.size() > 1 ? atoi(pos[1]) : 5);
    const std::string sfx = g_pinned ? "_pinned" : "";
    auto nm = [&](const char* base) { return std::string(base) + sfx; };

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "GPU: %s, SMs=%d, cc=%d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    Graph g;
    if (!load_graph(graph_path, g)) return 1;
    fprintf(stderr, "loaded n=%u m=%u max_out_deg=%u\n", g.n, g.m, g.max_out_degree);
    const uint64_t expected = cpu_triangles_serial(g);

    std::vector<uint32_t> t_edge, t_start;
    const double tb0 = now_ms();
    build_tasks(g, TASK_CHUNK, t_edge, t_start);
    const float task_build_ms = (float)(now_ms() - tb0);
    const uint32_t ntasks = (uint32_t)t_edge.size();
    fprintf(stderr, "v4 tasks=%u (chunk=%d), host task build %.2f ms\n", ntasks, TASK_CHUNK, task_build_ms);

    std::vector<uint32_t> perm;
    const double pb0 = now_ms();
    build_edge_order(g, perm);
    const float perm_build_ms = (float)(now_ms() - pb0);
    fprintf(stderr, "v9 edge order built, host time %.2f ms\n", perm_build_ms);

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    float h_rowptr = 0, h_col = 0, h_src = 0, h_te = 0, h_ts = 0, h_perm = 0;   // per-array copy times
    uint32_t* d_rowptr = upload(g.rowptr, &h_rowptr, e0, e1);
    uint32_t* d_col = upload(g.col, &h_col, e0, e1);
    uint32_t* d_src = upload(g.src, &h_src, e0, e1);
    uint32_t* d_tedge = upload(t_edge, &h_te, e0, e1);
    uint32_t* d_tstart = upload(t_start, &h_ts, e0, e1);
    uint32_t* d_perm = upload(perm, &h_perm, e0, e1);
    unsigned long long* d_total;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(unsigned long long)));

    const int block = 256;
    auto blocks_for_warps = [&](uint64_t warps) {
        uint64_t b = (warps * 32 + block - 1) / block;
        return (int)std::max<uint64_t>(1, std::min<uint64_t>(b, 1u << 20));
    };

    struct Variant { std::string name; float h2d; float host_prep; };
    bool ok = true;

    auto run = [&](const Variant& var, const std::function<void()>& launch) {
        unsigned long long result = 0;
        float best = 1e30f;
        for (int r = -1; r < repeats; r++) {           // r == -1: untimed warm-up
            CUDA_CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));
            CUDA_CHECK(cudaEventRecord(e0));
            launch();
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            CUDA_CHECK(cudaGetLastError());
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (r >= 0) best = std::min(best, ms);
            CUDA_CHECK(cudaMemcpy(&result, d_total, sizeof result, cudaMemcpyDeviceToHost));
            if (result != expected) {                  // check EVERY launch, not just the last
                fprintf(stderr, "MISMATCH in %s (launch %d): gpu=%llu cpu=%llu\n", var.name.c_str(), r,
                        (unsigned long long)result, (unsigned long long)expected);
                ok = false;
            }
        }
        printf("%s,%s,%u,%u,%llu,%.3f,%.3f,%.3f\n", var.name.c_str(), graph_path, g.n, g.m,
               (unsigned long long)result, best, var.h2d, best + var.h2d + var.host_prep);
        fflush(stdout);
    };

    int g1 = blocks_for_warps((g.m + 31) / 32 + 1);    // ~one thread per edge
    // Burn-in: keep the GPU busy for ~1.5 s (untimed) so the SM clock has ramped up before ANY timed run.
    // Without it the first variant (v1) can run at a lower clock than later ones: Nsight Compute at a fixed
    // clock showed v1, v5(G=2) and v7 need about the same number of cycles, yet wall-clock times differed 2x.
    if (!g_no_burnin) {
        const double t0 = now_ms();
        do {
            tc_edge_per_thread<<<g1, block>>>(d_rowptr, d_col, d_src, g.m, d_total);
            CUDA_CHECK(cudaDeviceSynchronize());
        } while (now_ms() - t0 < 1500.0);
        CUDA_CHECK(cudaGetLastError());
    }
    run({nm("gpu_v1_edge_per_thread"), h_rowptr + h_col + h_src, 0.f}, [&] {
        tc_edge_per_thread<<<g1, block>>>(d_rowptr, d_col, d_src, g.m, d_total);
    });
    int g2 = blocks_for_warps(g.m);
    run({nm("gpu_v2_warp_per_edge"), h_rowptr + h_col + h_src, 0.f}, [&] {
        tc_warp_per_edge<<<g2, block>>>(d_rowptr, d_col, d_src, g.m, d_total);
    });
    int g3 = (int)std::max<uint32_t>(1, std::min<uint32_t>(g.n, 1u << 20));
    run({nm("gpu_v3_block_smem"), h_rowptr + h_col, 0.f}, [&] {
        tc_block_smem<<<g3, block>>>(d_rowptr, d_col, g.n, d_total);
    });
    int g4 = blocks_for_warps(std::max<uint32_t>(ntasks, 1));
    run({nm("gpu_v4_task_warp"), h_rowptr + h_col + h_src + h_te + h_ts, task_build_ms}, [&] {
        tc_task_warp<<<g4, block>>>(d_rowptr, d_col, d_src, d_tedge, d_tstart, ntasks, d_total);
    });

    // v5: G threads per edge; total threads ~= m*G (G=32 would equal v2, so it is not repeated here).
    auto blocks_for_threads = [&](uint64_t threads) {
        uint64_t b = (threads + block - 1) / block;
        return (int)std::max<uint64_t>(1, std::min<uint64_t>(b, 1u << 20));
    };
    {
        int gb = blocks_for_threads((uint64_t)g.m * 2);
        run({nm("gpu_v5_g2_per_edge"), h_rowptr + h_col + h_src, 0.f},
            make_v5_launch<2>(gb, block, d_rowptr, d_col, d_src, g.m, d_total));
    }
    {
        int gb = blocks_for_threads((uint64_t)g.m * 4);
        run({nm("gpu_v5_g4_per_edge"), h_rowptr + h_col + h_src, 0.f},
            make_v5_launch<4>(gb, block, d_rowptr, d_col, d_src, g.m, d_total));
    }
    {
        int gb = blocks_for_threads((uint64_t)g.m * 8);
        run({nm("gpu_v5_g8_per_edge"), h_rowptr + h_col + h_src, 0.f},
            make_v5_launch<8>(gb, block, d_rowptr, d_col, d_src, g.m, d_total));
    }
    {
        int gb = blocks_for_threads((uint64_t)g.m * 16);
        run({nm("gpu_v5_g16_per_edge"), h_rowptr + h_col + h_src, 0.f},
            make_v5_launch<16>(gb, block, d_rowptr, d_col, d_src, g.m, d_total));
    }

    // v6: one thread per vertex (~n threads, like v3 capped at 1<<20 blocks). v7: one thread per edge like v1.
    {
        int gv6 = blocks_for_threads(g.n);
        run({nm("gpu_v6_vertex_per_thread"), h_rowptr + h_col, 0.f}, [&] {
            tc_vertex_per_thread<<<gv6, block>>>(d_rowptr, d_col, g.n, d_total);
        });
        run({nm("gpu_v7_edge_search_src"), h_rowptr + h_col, 0.f}, [&] {
            tc_edge_find_src<<<g1, block>>>(d_rowptr, d_col, g.n, g.m, d_total);
        });
    }

    // v8: one warp per vertex => n warps, 8 warps (256 threads) per block, which matches the static shared
    // table of V8_MAX_WARPS warps. v9: one thread per edge like v1, but through perm.
    static_assert(V8_MAX_WARPS * 32 == 256, "run() uses block = 256; V8_MAX_WARPS must be 8 for that block size");
    {
        int g8 = blocks_for_warps(std::max<uint32_t>(g.n, 1));
        run({nm("gpu_v8_warp_vertex_hash"), h_rowptr + h_col, 0.f}, [&] {
            tc_warp_vertex_hash<<<g8, block>>>(d_rowptr, d_col, g.n, d_total);
        });
        run({nm("gpu_v9_edge_sorted"), h_rowptr + h_col + h_src + h_perm, perm_build_ms}, [&] {
            tc_edge_sorted<<<g1, block>>>(d_rowptr, d_col, d_src, d_perm, g.m, d_total);
        });
    }

    if (!ok) return 1;
    fprintf(stderr, "all kernels verified against CPU count: %llu\n", (unsigned long long)expected);
    return 0;
}
