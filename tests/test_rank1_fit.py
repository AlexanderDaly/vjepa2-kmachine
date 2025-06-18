from multiplicative_fit import fit_rank1
import numpy as np


def test_rank1_fit():
    orbit_scores = { (0,0):4, (1,1):1 }
    a = fit_rank1(orbit_scores, n_symbols=2, k=2, lr=0.1, max_steps=2000)
    assert np.allclose(np.prod(a), 1.0, atol=1e-4)
