#!/usr/bin/env python3
"""Draw one stacked-bar chart per graph (>= 100000 edges) from results/results.csv.

Usage: make_charts.py [csv_path]
Output: results/chart_<graph>.png (overwritten).
Per GPU impl: kernel ms + host-to-device ms (+ host prep ms for v4 = gpu_total - kernel - h2d).
Serial and OpenMP CPU times are drawn as separate grey bars on the same linear axis, so the GPU
end-to-end bars can be compared with them directly. Best (minimum) run per impl is used.
"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_EDGES = 100000
# Okabe-Ito colour-blind-safe palette
C_KERNEL, C_H2D, C_PREP, C_CPU = "#0072B2", "#E69F00", "#CC79A7", "#7F7F7F"


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "results.csv")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("impl") or "").strip()]
    graphs = {}
    for r in rows:
        g = os.path.splitext(os.path.basename((r.get("graph") or "").strip()))[0]
        graphs.setdefault(g, []).append(r)
    outdir = os.path.dirname(path)
    for g, rs in sorted(graphs.items()):
        m = max((num(r.get("m")) or 0) for r in rs)
        if m < MIN_EDGES:
            continue
        best = {}
        for r in rs:
            t = num(r.get("time_ms"))
            if t is None:
                continue
            impl = r["impl"].strip()
            if impl not in best or t < num(best[impl]["time_ms"]):
                best[impl] = r
        gpu_names = {(r.get("gpu") or "").strip() for i, r in best.items() if i.startswith("gpu_")}
        gpu_names.discard("")
        gpu = ", ".join(sorted(gpu_names)) or "unknown GPU"
        gpu_impls = sorted(i for i in best if i.startswith("gpu_"))
        cpu_impls = [i for i in ["cpu_serial"] + sorted(i for i in best if i.startswith("cpu_omp")) if i in best]
        labels, kern, h2d, prep = [], [], [], []
        for i in gpu_impls:
            r = best[i]
            k, h, tot = num(r["time_ms"]), num(r.get("h2d_ms")) or 0.0, num(r.get("gpu_total_ms"))
            p = max(0.0, tot - k - h) if tot is not None else 0.0
            if p < 1e-9:
                p = 0.0
            labels.append(i.replace("gpu_", "")); kern.append(k); h2d.append(h); prep.append(p)
        n_gpu, n_cpu = len(gpu_impls), len(cpu_impls)
        fig, ax = plt.subplots(figsize=(10, 0.45 * (n_gpu + n_cpu) + 2.2))
        ypos = list(range(n_gpu + n_cpu))[::-1]            # first entry on top
        for y, k, h, p in zip(ypos[:n_gpu], kern, h2d, prep):
            ax.barh(y, k, color=C_KERNEL, label="kernel")
            ax.barh(y, h, left=k, color=C_H2D, label="host-to-device copy")
            if p:
                ax.barh(y, p, left=k + h, color=C_PREP, label="host prep (task build)")
            ax.text(k + h + p, y, f" {k + h + p:.1f}", va="center", fontsize=8)
        for y, i in zip(ypos[n_gpu:], cpu_impls):
            t = num(best[i]["time_ms"])
            ax.barh(y, t, color=C_CPU, label="CPU time")
            ax.text(t, y, f" {t:.1f}", va="center", fontsize=8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels + cpu_impls, fontsize=8)
        ax.set_xlabel("time (ms, lower is better; GPU bars = kernel + transfer [+ host prep], CPU bars = total)")
        ax.set_title(f"{g}: GPU end-to-end vs CPU ({gpu}; m={int(m):,} edges)", fontsize=10)
        h_, l_ = ax.get_legend_handles_labels()
        seen = dict(zip(l_[::-1], h_[::-1]))
        ax.legend(seen.values(), seen.keys(), loc="lower right", fontsize=8, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.08)
        fig.tight_layout()
        out = os.path.join(outdir, f"chart_{g}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
