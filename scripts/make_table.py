#!/usr/bin/env python3
"""Turn results/results.csv into Markdown tables (stdlib only).

Usage: make_table.py [csv_path] [--write-readme]

Per (impl, graph, gpu) the best (minimum) time over all appended runs is used.
CPU rows are not tied to a GPU, so they are merged across the gpu column.
Exit code 1 if any impl's triangle count differs from cpu_serial on the same graph.
"""
import csv, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


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


def build(rows):
    """Returns (markdown, mismatch_found, gpu_names)."""
    graphs = {}
    gpus = []
    for r in rows:
        g = os.path.splitext(os.path.basename((r.get("graph") or "").strip()))[0]
        impl = r["impl"].strip()
        gpu = (r.get("gpu") or "").strip().strip('"')
        is_cpu = impl.startswith("cpu_")
        if not is_cpu and gpu and gpu not in gpus:
            gpus.append(gpu)
        key = (impl, "" if is_cpu else gpu)
        graphs.setdefault(g, {}).setdefault(key, []).append(r)

    out, mismatch = [], False
    for g in sorted(graphs):
        groups = graphs[g]
        best = {}
        for key, rs in groups.items():
            timed = [r for r in rs if num(r.get("time_ms")) is not None]
            if not timed:
                continue
            b = min(timed, key=lambda r: num(r["time_ms"]))
            tris = {(r.get("triangles") or "").strip() for r in rs}
            best[key] = (b, tris)
        serial = [v for (impl, _), v in best.items() if impl == "cpu_serial"]
        ref = serial[0][0]["triangles"].strip() if serial else None
        ser_t = num(serial[0][0]["time_ms"]) if serial else None
        omp = [(num(v[0]["time_ms"]), impl) for (impl, _), v in best.items() if impl.startswith("cpu_omp")]
        omp_t, omp_name = min(omp) if omp else (None, None)

        first = next(iter(best.values()))[0] if best else {}
        out.append(f"### {g}  (n={first.get('n','?')}, m={first.get('m','?')})\n")
        out.append(f"Speedup columns use kernel-only time for GPU rows. OMP reference: {omp_name or 'none in CSV'}.\n")
        out.append("| impl | GPU | triangles | kernel/CPU ms | end-to-end ms (GPU) | speedup vs cpu_serial | speedup vs OMP | check |")
        out.append("|---|---|---|---|---|---|---|---|")

        def order(k):
            impl = k[0]
            return (0 if impl == "cpu_serial" else 1 if impl.startswith("cpu_omp") else 2, impl, k[1])

        for key in sorted(best, key=order):
            impl, gpu = key
            b, tris = best[key]
            t = num(b["time_ms"])
            e2e = num(b.get("gpu_total_ms")) if not impl.startswith("cpu_") else None
            if ref is None:
                check = "no cpu_serial row"
            elif tris != {ref}:
                check = "COUNT MISMATCH"
                mismatch = True
            else:
                check = "ok"
            bad = check == "COUNT MISMATCH"
            sp_s = "-" if bad or not ser_t or not t or impl == "cpu_serial" else f"{ser_t / t:.2f}x"
            sp_o = "-" if bad or not omp_t or not t or impl.startswith("cpu_") else f"{omp_t / t:.2f}x"
            out.append(f"| {impl} | {gpu or '-'} | {b['triangles'].strip()} | {fmt(t)} | {fmt(e2e)} | {sp_s} | {sp_o} | {check} |")
        out.append("")
    if not out:
        out = ["Not yet measured. Run scripts/run_bench.sh on a GPU, then `python3 scripts/make_table.py --write-readme`.", ""]
    return "\n".join(out), mismatch, gpus


def write_readme(block):
    p = os.path.join(ROOT, "README.md")
    with open(p) as f:
        s = f.read()
    if START not in s or END not in s or s.index(START) > s.index(END):
        sys.exit("README.md is missing the RESULTS markers")
    a, b = s.index(START) + len(START), s.index(END)
    with open(p, "w") as f:
        f.write(s[:a] + "\n" + block.rstrip() + "\n" + s[b:])
    print("README.md updated")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else os.path.join(ROOT, "results", "results.csv")
    rows = load(path)
    if not rows:
        print(f"No results found in {path} (missing or empty). Run scripts/run_bench.sh on a GPU first.")
    md, mismatch, gpus = build(rows)
    gpu_line = ("GPU(s) in results: " + "; ".join(gpus)) if gpus else "GPU(s) in results: none yet"
    full = md.rstrip() + "\n\n" + gpu_line + "\n"
    if mismatch:
        full += "\n**COUNT MISMATCH found: timings for those rows are not valid.**\n"
    print(full)
    if "--write-readme" in sys.argv:
        write_readme(full)
    if mismatch:
        print("COUNT MISMATCH", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
