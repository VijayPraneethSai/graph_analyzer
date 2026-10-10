#!/usr/bin/env python3
"""Effective-bandwidth estimate per GPU kernel from results/results.csv (stdlib only).

Usage: bandwidth_estimate.py [csv_path]

effective GB/s = (minimum bytes the kernel must read once) / kernel time.
  v3, v6, v7      : rowptr + col                = 4*(n+1) + 4*m
  v1, v2, v5-family: rowptr + col + src         = 4*(n+1) + 4*m + 4*m
  v4 is skipped (its task arrays are not in the CSV).
"""
import csv, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Published spec value (GDDR6, 320 GB/s). Only meaningful if the GPU in the CSV is a Tesla T4;
# for any other GPU the percentage is printed as n/a.
T4_PEAK_GBPS = 320.0

HEADER = ("Effective bandwidth of the MINIMUM data, not measured memory traffic (lists are re-read by "
          "many edges); an estimate, not a profiler measurement.")


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def min_bytes(impl, n, m):
    """Returns bytes, or None if the impl is skipped / unknown."""
    base = impl[:-len("_pinned")] if impl.endswith("_pinned") else impl
    if base.startswith("gpu_v4"):
        return None
    if base.startswith(("gpu_v3", "gpu_v6", "gpu_v7")):
        return 4 * (n + 1) + 4 * m
    if base.startswith(("gpu_v1", "gpu_v2", "gpu_v5")):
        return 4 * (n + 1) + 8 * m
    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "results.csv")
    if not os.path.isfile(path):
        sys.exit(f"No such file: {path}")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("impl") or "").strip()]
    # best (minimum kernel time) per (graph, impl, gpu)
    best, skipped = {}, set()
    for r in rows:
        impl = r["impl"].strip()
        if not impl.startswith("gpu_"):
            continue
        t, n, m = num(r.get("time_ms")), num(r.get("n")), num(r.get("m"))
        if t is None or t <= 0 or n is None or m is None:
            continue
        gname = os.path.splitext(os.path.basename((r.get("graph") or "").strip()))[0]
        gpu = (r.get("gpu") or "").strip().strip('"')
        b = min_bytes(impl, n, m)
        if b is None:
            skipped.add(impl)
            continue
        key = (gname, gpu, impl)
        if key not in best or t < best[key][0]:
            best[key] = (t, b)
    print(HEADER + "\n")
    if skipped:
        print("Skipped (task arrays not in CSV): " + ", ".join(sorted(skipped)) + "\n")
    for gname, gpu in sorted({(k[0], k[1]) for k in best}):
        is_t4 = "tesla t4" in gpu.lower()
        print(f"### {gname}  ({gpu or 'unknown GPU'})\n")
        print("| impl | kernel ms | min MB | effective GB/s | % of T4 peak (320 GB/s) |")
        print("|---|---|---|---|---|")
        for (g2, p2, impl), (t, b) in sorted(best.items()):
            if (g2, p2) != (gname, gpu):
                continue
            gbps = b / (t * 1e-3) / 1e9
            pct = f"{100 * gbps / T4_PEAK_GBPS:.1f}%" if is_t4 else "n/a"
            print(f"| {impl} | {t:.3f} | {b / 1e6:.2f} | {gbps:.1f} | {pct} |")
        print()


if __name__ == "__main__":
    main()
