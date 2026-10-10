#!/usr/bin/env python3
"""Mutation check for the stage-6 tests (no GPU needed).
Copies src/ and emu/ into a temp dir, applies ONE deliberate bug (the pattern must occur exactly once), builds
emu/emu_dsg.cpp and runs `emu_dsg --tiny`. A mutant is CAUGHT if the build works and the run does not end in "ALL OK"
(failed checks, crash, or timeout); SURVIVED if the tests still print "ALL OK". Build errors are reported as INVALID.
usage: scripts/mutation_check.py [name ...]     (default: all mutants; the unmutated baseline is run first and must pass)
"""
import concurrent.futures as cf, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (name, file, old text, new text, comment)
M = [
 ("core_floor_gt_to_ge", "src/dsg.h", "if (deg[u] > core[v]) {", "if (deg[u] >= core[v]) {", "Alg 3 line 8: > flipped to >="),
 ("core_drop_decrement", "src/dsg.h", "                        deg[u]--;\n", "", "Alg 3: the degree decrement is dropped"),
 ("kernel_peel_le_to_lt", "src/kernels.cuh", "(unsigned long long)tri[v] <= k)", "(unsigned long long)tri[v] < k)", "core_peel_level: <= flipped to <"),
 ("kernel_peel_k_ignores_prev", "src/kernels.cuh", "const unsigned long long k = m > k_in ? m : k_in;", "const unsigned long long k = m;", "core_peel_level: k = max(k_in,min) -> min"),
 ("kernel_min_becomes_max", "src/kernels.cuh", "if (alive[v] && tri[v] < local) local = tri[v];", "if (alive[v] && tri[v] > local) local = tri[v];", "core_min_alive_tri computes a max-like value"),
 ("loop_level_reset", "src/dsg.h", "        k = k_new;\n        r.rounds = round + 1;", "        k = 0;\n        r.rounds = round + 1;", "host loop forgets the level between rounds"),
 ("flow_sink_cap_2a", "src/dsg.h", "D.cap[sink_arc[v]] = 3 * a;", "D.cap[sink_arc[v]] = 2 * a;", "v->t capacity alpha*|V_Psi| uses 2 instead of 3"),
 ("flow_arc_cap_2Q", "src/dsg.h", "D.add_edge(2 + (int)third[k], pn, (i64)Q);", "D.add_edge(2 + (int)third[k], pn, (i64)(2 * Q));", "v->psi capacity 2 instead of 1"),
 ("dinic_no_reverse_cap", "src/dsg.h", "for (int e : path) { cap[e] -= f; cap[e ^ 1] += f; }", "for (int e : path) { cap[e] -= f; }", "Dinic forgets residual (reverse) capacity"),
 ("exact_stops_early", "src/dsg.h", "    while (hi - lo >= 2) {\n        const i64 mid = lo + (hi - lo) / 2;\n        std::vector<uint32_t> S = net.solve(mid);\n        r.flow_calls++;\n        if (S.empty()) { hi = mid; continue; }\n        lo = mid;\n        const Dens d{net.count_in(S), S.size()};",
  "    while (hi - lo >= 40) {\n        const i64 mid = lo + (hi - lo) / 2;\n        std::vector<uint32_t> S = net.solve(mid);\n        r.flow_calls++;\n        if (S.empty()) { hi = mid; continue; }\n        lo = mid;\n        const Dens d{net.count_in(S), S.size()};", "Exact's binary search stops far too early"),
 ("coreexact_k1_too_big", "src/dsg.h", "const uint64_t k1 = dens_ceil(best);", "const uint64_t k1 = dens_ceil(best) + 1;", "Pruning1 uses ceil(rho')+1 (prunes too much)"),
 ("coreexact_shrink_too_big", "src/dsg.h", "const uint64_t knew = ((uint64_t)mid + Q - 1) / Q;", "const uint64_t knew = ((uint64_t)mid + 2 * Q - 1) / Q;", "Line 17 shrinks to the (ceil(alpha)+1)-core"),
 ("coreexact_comp_strict", "src/dsg.h", "if (core[y] >= k && label[y] == UINT32_MAX)", "if (core[y] > k && label[y] == UINT32_MAX)", "components use core > k"),
 ("coreexact_no_u_reset_hi", "src/dsg.h", "i64 hi = (i64)((u128)kmax * Q);", "i64 hi = (i64)((u128)(kmax > 1 ? kmax - 1 : 1) * Q);", "upper bound u = kmax-1 (too small)"),
 ("coreexact_skip_initial_candidate", "src/dsg.h", "        if (!consider(S, lo)) return r;\n        while", "        while", "initial check (lines 7-9) no longer records its subgraph"),
 ("prune_core_gt", "src/dsg.h", "for (uint32_t v = 0; v < g.n; v++) keep[v] = core[v] >= k;\n    induced_subgraph(g, keep, out, old_id);", "for (uint32_t v = 0; v < g.n; v++) keep[v] = core[v] > k;\n    induced_subgraph(g, keep, out, old_id);", "pruning keeps core > ceil(lb) instead of >="),
 ("gpp_loads_not_accumulated", "src/dsg.h", "next_load[v] = load[v] + cur[v];", "next_load[v] = load[v];", "Greedy++ loads never grow (degenerates to greedy)"),
 ("gpp_wrong_pass_best", "src/dsg.h", "r.per_pass.push_back(best);", "r.per_pass.push_back(Dens{T0, g.n});", "per_pass records the wrong value"),
 ("dens_cmp_le", "src/dsg.h", "return l < r ? -1 : (l > r ? 1 : 0);", "return l <= r ? -1 : (l > r ? 1 : 0);", "equal fractions compare as smaller"),
 ("induced_edge_or", "src/dsg.h", "if (keep[a] && keep[b]) edges.emplace_back(new_id[a], new_id[b]);", "if (keep[a] || keep[b]) edges.emplace_back(new_id[a], new_id[b]);", "induced_subgraph keeps edges with ONE kept endpoint"),
 ("builder_orientation_flipped", "src/graph.h", "return deg[a] != deg[b] ? deg[a] < deg[b] : a < b;", "return deg[a] != deg[b] ? deg[a] > deg[b] : a < b;", "orientation from the larger-degree endpoint (still a valid DAG, wrong rule)"),
 ("greedy_tie_keeps_later", "src/dsg.h", "if (dens_gt(Dens{Tcur, nalive}, best)) { best", "if (dens_cmp(Dens{Tcur, nalive}, best) >= 0 && Tcur > 0) { best", "ties replace the best (same density, smaller set) - expected to be an EQUIVALENT mutant for density checks"),
]

def run(m, tmp_root):
    name, f, old, new, cmt = m
    d = tempfile.mkdtemp(dir=tmp_root)
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(d, "src"))
    shutil.copytree(os.path.join(ROOT, "emu"), os.path.join(d, "emu"))
    os.makedirs(os.path.join(d, "bin"))
    if old is not None:
        p = os.path.join(d, f)
        s = open(p).read()
        if s.count(old) != 1:
            return name, "INVALID", f"pattern occurs {s.count(old)} times", cmt
        open(p, "w").write(s.replace(old, new))
    b = subprocess.run(["g++", "-O2", "-std=c++20", "-pthread", "-fopenmp", "emu/emu_dsg.cpp", "-o", "bin/emu_dsg"], cwd=d, capture_output=True, text=True)
    if b.returncode:
        return name, "INVALID", "build error: " + b.stderr.strip().splitlines()[0][:100], cmt
    try:
        r = subprocess.run(["./bin/emu_dsg", "--tiny"], cwd=d, capture_output=True, text=True, timeout=600)
        out = r.stdout.strip().splitlines()
        ok = r.returncode == 0 and out and out[-1] == "ALL OK"
        first = next((l.strip() for l in out if "FAIL" in l), "") if not ok else ""
        status = "SURVIVED" if ok else ("CAUGHT" if r.returncode in (0, 1) else f"CAUGHT(crash rc={r.returncode})")
        detail = first[:110] if first else ("" if ok else f"rc={r.returncode}")
    except subprocess.TimeoutExpired:
        status, detail = "CAUGHT(timeout)", "timeout"
    shutil.rmtree(d, ignore_errors=True)
    return name, status, detail, cmt

if __name__ == "__main__":
    want = set(sys.argv[1:])
    muts = [m for m in M if not want or m[0] in want]
    tmp_root = tempfile.mkdtemp()
    base = run(("BASELINE", "src/dsg.h", None, None, "unmutated"), tmp_root)
    print(f"{base[0]:34s} {base[1]:9s} {base[2]}", flush=True)
    if base[1] != "SURVIVED":
        print("baseline does not pass - aborting"); sys.exit(1)
    res = []
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        for r in ex.map(lambda m: run(m, tmp_root), muts):
            res.append(r)
            print(f"{r[0]:34s} {r[1]:16s} {r[2]}   [{r[3]}]", flush=True)
    c = sum(r[1].startswith("CAUGHT") for r in res); s = [r[0] for r in res if r[1] == "SURVIVED"]; i = [r[0] for r in res if r[1] == "INVALID"]
    print(f"\n{c} caught, {len(s)} survived {s}, {len(i)} invalid {i} (of {len(res)})")
    shutil.rmtree(tmp_root, ignore_errors=True)
