#!/usr/bin/env python3
"""Small random graph + brute-force triangle count, to sanity check the loader and kernels."""
import os, itertools, random, sys
n, p, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
out = sys.argv[4]
random.seed(seed)
edges = set()
for u in range(n):
    for v in range(u + 1, n):
        if random.random() < p:
            edges.add((u, v))
adj = {i: set() for i in range(n)}
for u, v in edges:
    adj[u].add(v); adj[v].add(u)
tri = sum(1 for u, v in edges for w in adj[u] & adj[v] if w > v)
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "w") as f:
    f.write("# synthetic\n")
    for u, v in sorted(edges):
        f.write(f"{u}\t{v}\n")
        if random.random() < 0.3:      # duplicate reversed edge to exercise dedup
            f.write(f"{v}\t{u}\n")
    f.write("5\t5\n")                   # self loop
print(tri)
