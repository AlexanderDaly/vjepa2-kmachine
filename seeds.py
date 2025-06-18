from random import sample
from itertools import product
import math


def default_seed_sample(km, n_seeds=128):
    """Pick heuristic seeds with maximal cost variance
    + symbol entropy to accelerate GAP orbit discovery."""
    V, k = range(km.n_symbols), km.k
    # rough stratified sampling
    bag = list(product(V, repeat=k))
    cand = sample(bag, min(len(bag), 2048))
    mean = sum(km.cost(x) for x in cand) / len(cand)
    std = math.sqrt(sum((km.cost(x)-mean)**2 for x in cand) / len(cand))
    seeds = [
        x for x in cand if abs(km.cost(x) - mean) > std * 0.75
    ]
    return seeds[:n_seeds] if len(seeds) >= n_seeds else cand[:n_seeds]
