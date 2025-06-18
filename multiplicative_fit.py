import numpy as np
from typing import Dict, List, Tuple

String = Tuple[int, ...]


def fit_rank1(
    orbit_scores: Dict[String, float],
    n_symbols: int,
    k: int,
    lr=0.1,
    max_steps=5000,
    tol=1e-6,
    patience=100,
):
    """Stochastic gradient descent (log-space) for multiplicative model:

        f(x)  ≈  C · Π_j a[x_j]

    Returns normalised positive vector  a  (np.ndarray, shape [n_symbols]).
    """
    a = np.zeros(n_symbols, dtype=float)
    best = np.inf
    no_improve = 0

    keys = list(orbit_scores.keys())
    vals = np.array([orbit_scores[x] for x in keys])

    for step in range(max_steps):
        idx = np.random.randint(len(keys))
        x, s = keys[idx], vals[idx]

        log_pred = sum(a[i] for i in x)
        pred = np.exp(log_pred)

        loss = (pred - s) ** 2
        if loss < best - tol:
            best, no_improve = loss, 0
        else:
            no_improve += 1
            if no_improve > patience:
                break

        for i in x:
            a[i] -= lr * 2 * (pred - s) * pred

    a = np.exp(a - a.mean())
    return a
