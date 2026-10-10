#!/usr/bin/env python3
"""Preferential-attachment (skewed, hub-heavy) graph + brute-force triangle count.
usage: gen_powerlaw.py n k seed out.txt   (each new vertex attaches to k existing vertices)"""
import os, random, sys
n, k, seed, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
random.seed(seed)
targets = list(range(k))
repeated = []
edges = set()
for v in range(k, n):
    chosen = set()
    while len(chosen) < k:
        chosen.add(random.choice(repeated) if repeated and random.random() < 0.9 else random.randrange(v))
    for t in chosen:
        edges.add((min(v, t), max(v, t)))
        repeated += [v, t]
adj = {i: set() for i in range(n)}
for u, v in edges:
    adj[u].add(v); adj[v].add(u)
tri = sum(1 for u, v in edges for w in adj[u] & adj[v] if w > v)
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "w") as f:
    f.write("# power-law synthetic\n")
    for u, v in sorted(edges):
        f.write(f"{u}\t{v}\n")
print(tri)
