// cuda_shim.h - minimal CPU emulation of the CUDA features used by src/kernels.cuh.
// Each CUDA thread is a real std::thread; __syncthreads is a block barrier, warp shuffles use
// per-warp barriers + a slot array. Blocks run one after another. Shared memory = function statics
// (valid because blocks are serialised). This checks kernel LOGIC (indexing, strides, reductions,
// barriers), not GPU performance or memory-model behaviour.
#pragma once
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

struct dim3 { unsigned x = 1, y = 1, z = 1; dim3(unsigned x_ = 1) : x(x_) {} };
static thread_local dim3 threadIdx, blockIdx, blockDim, gridDim;
static thread_local std::barrier<>* g_block_bar = nullptr;
static thread_local std::barrier<>* g_warp_bar = nullptr;
static thread_local unsigned long long* g_slots = nullptr;   // one slot per thread of the block

#define __global__
#define __device__
#define __forceinline__ inline
#define __restrict__ __restrict
#define __shared__ static

inline void __syncwarp(unsigned = 0xffffffffu) { g_warp_bar->arrive_and_wait(); }
inline void __syncthreads() { g_block_bar->arrive_and_wait(); }

template <class T>
inline T __shfl_down_sync(unsigned, T v, int delta) {
    unsigned lane = threadIdx.x & 31, base = threadIdx.x & ~31u;
    unsigned long long bits = 0; std::memcpy(&bits, &v, sizeof(T));
    g_slots[threadIdx.x] = bits;
    g_warp_bar->arrive_and_wait();
    T r = v;
    if (lane + delta < 32) { unsigned long long b = g_slots[base + lane + delta]; std::memcpy(&r, &b, sizeof(T)); }
    g_warp_bar->arrive_and_wait();
    return r;
}

inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) {
    return __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
}

inline unsigned atomicAdd(unsigned* p, unsigned v) {   // uint32 counters (tri[] in tc_per_vertex_alive)
    return __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
}

inline unsigned atomicSub(unsigned* p, unsigned v) {   // uint32 decrement (core_update_frontier)
    return __atomic_fetch_sub(p, v, __ATOMIC_RELAXED);
}

inline unsigned atomicMin(unsigned* p, unsigned v) {   // uint32 minimum (core_min_alive_tri)
    unsigned old = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (v < old && !__atomic_compare_exchange_n(p, &old, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    return old;
}

// atomicCAS on a (shared or global) uint32: returns the old value, stores `val` only if *p == cmp.
inline uint32_t atomicCAS(uint32_t* p, uint32_t cmp, uint32_t val) {
    __atomic_compare_exchange_n(p, &cmp, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return cmp;                                        // on failure cmp now holds the observed value
}

// Runs `kernel_call` for every thread of a grid x block launch (blocks sequentially).
inline void emu_launch(unsigned grid, unsigned block, const std::function<void()>& kernel_call) {
    for (unsigned b = 0; b < grid; b++) {
        std::barrier<> block_bar(block);
        std::vector<std::unique_ptr<std::barrier<>>> warp_bars;
        for (unsigned w = 0; w < block / 32; w++) warp_bars.emplace_back(new std::barrier<>(32));
        std::vector<unsigned long long> slots(block);
        std::vector<std::thread> ts;
        for (unsigned t = 0; t < block; t++)
            ts.emplace_back([&, t] {
                threadIdx.x = t; blockIdx.x = b; blockDim.x = block; gridDim.x = grid;
                g_block_bar = &block_bar; g_warp_bar = warp_bars[t / 32].get(); g_slots = slots.data();
                kernel_call();
            });
        for (auto& th : ts) th.join();
    }
}
