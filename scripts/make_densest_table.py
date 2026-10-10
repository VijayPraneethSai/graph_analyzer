#!/usr/bin/env python3
"""Turn results/results_densest.csv into Markdown tables (stdlib only). Does not touch results.csv.

Usage: make_densest_table.py [csv_path] [--write-readme]

Columns of the CSV: impl,graph,n,m,best_triangles,best_vertices,density,rounds,time_ms,h2d_ms,total_ms,gpu,date
Per (graph, impl, gpu) the best (minimum) time_ms over all appended runs is used; CPU rows are not tied to a GPU.
Speedups are ratios of those best times:
  kernel-only = cpu time_ms / gpu time_ms          (gpu time_ms = all peeling rounds incl. per-round host syncs, no uploads)
  end-to-end  = cpu total_ms / gpu total_ms        (gpu total_ms = time_ms + host-to-device upload of rowptr, col, src)
Density ratio = gpu density / greedy density, computed from the integer pairs (triangles, vertices), not from the
printed 6-decimal value. The CSV does not store eps: keep ONE eps per results file. Exit code 1 if the GPU result
(triangles, vertices) differs from cpu_batch_peel on the same graph (should be impossible: gpu_densest verifies
bit-exactness before printing).
--write-readme replaces the text between <!-- DENSEST:START --> and <!-- DENSEST:END --> in README.md.
"""
import csv, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START, END = "<!-- DENSEST:START -->", "<!-- DENSEST:END -->"


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("impl") or "").strip()]


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def ratio(a, b):
    return None if a is None or b is None or b == 0 else a / b


def build(rows):
    """Returns (markdown, mismatch_found, gpu_names)."""
    graphs, gpus = {}, []
    for r in rows:
        g = os.path.splitext(os.path.basename((r.get("graph") or "").strip()))[0]
        impl = r["impl"].strip()
        gpu = (r.get("gpu") or "").strip().strip('"')
        is_cpu = impl.startswith("cpu_")
        if not is_cpu and gpu and gpu not in gpus:
            gpus.append(gpu)
        graphs.setdefault(g, {}).setdefault((impl, "" if is_cpu else gpu), []).append(r)

    detail, summary, mismatch = [], [], False
    for g in sorted(graphs):
        best = {}
        for key, rs in graphs[g].items():
            timed = [r for r in rs if num(r.get("time_ms")) is not None]
            if timed:
                best[key] = min(timed, key=lambda r: num(r["time_ms"]))
        def cpu(impl):
            return best.get((impl, ""))
        ref = cpu("cpu_batch_peel")
        greedy = cpu("cpu_greedy_peel")
        omp = cpu("cpu_batch_peel_omp")
        n = next(iter(best.values()))["n"] if best else "?"
        m = next(iter(best.values()))["m"] if best else "?"
        detail.append(f"**{g}** (n={n}, m={m})\n")
        detail.append("| impl | GPU | best triangles | best vertices | density | rounds | time ms | h2d ms | total ms |")
        detail.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for (impl, gpu) in sorted(best):
            b = best[(impl, gpu)]
            flag = ""
            if ref is not None and impl.startswith("gpu_") and (b["best_triangles"], b["best_vertices"]) != (ref["best_triangles"], ref["best_vertices"]):
                flag, mismatch = " **MISMATCH vs cpu_batch_peel**", True
            detail.append(f"| {impl}{flag} | {gpu or '-'} | {b['best_triangles']} | {b['best_vertices']} | {fmt(num(b['density']), 6)} | "
                          f"{b['rounds']} | {fmt(num(b['time_ms']))} | {fmt(num(b['h2d_ms']))} | {fmt(num(b['total_ms']))} |")
        detail.append("")
        for (impl, gpu), b in sorted(best.items()):
            if not impl.startswith("gpu_batch_peel"):
                continue
            gk, gt = num(b["time_ms"]), num(b["total_ms"])
            gden = None
            if num(b["best_vertices"]):
                gden = num(b["best_triangles"]) / num(b["best_vertices"])
            qual = None
            if greedy is not None and num(greedy["best_vertices"]) and num(greedy["best_triangles"]):
                qual = gden / (num(greedy["best_triangles"]) / num(greedy["best_vertices"])) if gden is not None else None
            def sp(c, field):
                return fmt(ratio(num(c[field]), gk if field == "time_ms" else gt), 2) if c is not None else "-"
            summary.append(f"| {g} | {impl} | {gpu or '-'} | {sp(ref, 'time_ms')} | {sp(omp, 'time_ms')} | {sp(greedy, 'time_ms')} | "
                           f"{sp(ref, 'total_ms')} | {sp(omp, 'total_ms')} | {sp(greedy, 'total_ms')} | {fmt(qual, 4)} |")
    out = []
    if summary:
        out.append("Speedup of `gpu_batch_peel` (x, >1 = GPU faster; best-of times) and density relative to the sequential greedy baseline:\n")
        out.append("| graph | impl | GPU | kernel-only vs cpu_batch_peel | vs cpu_batch_peel_omp | vs cpu_greedy_peel | end-to-end vs cpu_batch_peel | vs cpu_batch_peel_omp | vs cpu_greedy_peel | density gpu/greedy |")
        out.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        out += summary
        out.append("")
    out += detail
    return "\n".join(out), mismatch, gpus


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else os.path.join(ROOT, "results", "results_densest.csv")
    rows = load(path)
    if not rows:
        print(f"no rows in {path}", file=sys.stderr)
        return 1
    md, mismatch, gpus = build(rows)
    if "--write-readme" in sys.argv:
        rp = os.path.join(ROOT, "README.md")
        text = open(rp).read()
        if START not in text or END not in text:
            print(f"README.md lacks the {START} / {END} markers", file=sys.stderr)
            return 1
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        note = f"_GPU: {', '.join(gpus) if gpus else 'n/a'}. Source: results/results_densest.csv._\n\n"
        open(rp, "w").write(head + START + "\n" + note + md + "\n" + END + tail)
        print("README.md updated")
    else:
        print(md)
    return 1 if mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
