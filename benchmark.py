"""
Benchmark canonical vs. GAP orbit enumeration on a 4070 Ti.

Run:
    python benchmark.py --n 6 --k 10 --gap
"""
import argparse, time, tracemalloc, psutil
from kabbalistic_machine import KabbalisticMachine


def profile(n, k, use_gap):
    km = KabbalisticMachine(n, k, cost=lambda x: sum(x), use_gap=use_gap)
    start = time.perf_counter()
    tracemalloc.start()
    km.analyse()
    mem = tracemalloc.get_traced_memory()[1] / 1024**2
    tracemalloc.stop()
    secs = time.perf_counter() - start
    print(
        f"{'GAP' if use_gap else 'Canonical'}  n={n}, k={k}  "
        f"{secs:7.2f}s,  peak\xa0{mem:8.1f}\u202fMB,  orbits={len(km._orbit_scores)}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--gap", action="store_true")
    args = ap.parse_args()
    profile(args.n, args.k, args.gap)
