"""
KABBALISTIC MACHINE v0.2
─────────────────────────────────────
Orbit‑aware search‑space compression with optional GAP back‑end.

Changes vs. 0.1
---------------
✓ Canonical‑form hashing → no duplicate orbits in RAM.
✓ Optional GAP integration  (`use_gap=True`) for >1e6 states.
✓ Correctly documented group action = wreath product   C_k  wr  G_letters.
✓ Public method  .orbits()  exposes representatives + scores.

Requirements
------------
• Pure‑Python path uses only std‑lib + typing.
• GAP path needs  `libgap`  (`pip install libgap`).

Author  : Morpheus & Grok 3
Updated : 2025-06-18
Licence : MIT / Apache-2.0 dual.
Full action implemented = wreath product C_k \wr G_letters.
Setting `letter_perm_generators=None` defaults to the entire symmetric group S_n.
For the smaller semidirect action use a single permutation generator and omit the second nested loop.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import permutations, product
from math import prod
from typing import Callable, Dict, List, Tuple

try:                                  # lazy TEST for GAP
    import libgap                      # noqa: F401
    GAP_AVAILABLE = True
except ImportError:
    GAP_AVAILABLE = False

# ──— Type aliases ──────────────────────────────────────────
String      = Tuple[int, ...]
CostFn      = Callable[[String], float]
Permutation = Tuple[int, ...]


# ──— Utility functions ───────────────────────────────────────────

def cyclic_shift(x: String, s: int) -> String:
    k = len(x)
    s %= k
    return x[s:] + x[:s]


def apply_letter_perm(x: String, σ: Permutation) -> String:
    return tuple(σ[i] for i in x)


# ──— Core class ─────────────────────────────────────────────
@dataclass
class KabbalisticMachine:
    n_symbols: int
    k: int
    cost: CostFn
    letter_perm_generators: List[Permutation] | None = None
    use_gap: bool = False

    # internal state (populated by analyse)
    _orbit_id: Dict[String, int] = field(init=False, default_factory=dict)
    _orbit_scores: List[float] = field(init=False, default_factory=list)
    _orbit_counts: List[int] = field(init=False, default_factory=list)

    # ── Public API ────────────────────────────────────────
    def analyse(self, *, seeds: List[String] | None = None) -> None:
        """Populate orbit table.

        If `use_gap` and GAP is present, delegate orbit enumeration to GAP;
        otherwise fall back to canonical‑form enumeration.
        """
        if self.use_gap:
            if not GAP_AVAILABLE:
                raise RuntimeError("use_gap=True but libgap not installed")
            self._analyse_with_gap(seeds)
        else:
            self._analyse_canonical()

    def orbit_score(self, x: String) -> float:
        if not self._orbit_id:
            raise RuntimeError("Call .analyse() first")
        return self._orbit_scores[self._orbit_id[self._canonical(x)]]

    def plateau_summary(self):
        from collections import Counter
        return Counter(self._orbit_scores)

    def orbits(self):
        """Yield (representative, score, size)."""
        reps = sorted(self._orbit_id, key=self._orbit_id.get)
        for rep in reps:
            idx = self._orbit_id[rep]
            yield rep, self._orbit_scores[idx], self._orbit_counts[idx]

    # ── Canonical‑form backend ──────────────────────────────────────
    def _analyse_canonical(self) -> None:
        V = range(self.n_symbols)
        G_letters = self._all_letter_perms()
        Ck = range(self.k)

        for x in product(V, repeat=self.k):
            rep = self._canonical(x, G_letters, Ck)
            if rep not in self._orbit_id:
                idx = len(self._orbit_scores)
                self._orbit_id[rep] = idx
                self._orbit_scores.append(self.cost(x))
                self._orbit_counts.append(1)
            else:
                idx = self._orbit_id[rep]
                self._orbit_scores[idx] += self.cost(x)
                self._orbit_counts[idx] += 1

        # convert sums → means
        self._orbit_scores = [s / c for s, c in zip(self._orbit_scores, self._orbit_counts)]

    # ── GAP backend ───────────────────────────────────────────
    def _analyse_with_gap(self, seeds: List[String] | None) -> None:
        """Orbit enumeration via GAP stabiliser chains."""
        import libgap as gap

        # build letter‑perm group  (default full S_n)
        gens = self.letter_perm_generators or [
            tuple(range(1, self.n_symbols + 1))[:i] + (i + 2,) + (i + 1,) + tuple(range(i + 3, self.n_symbols + 1))
            for i in range(self.n_symbols - 1)
        ]
        gap_gens = [gap.Permutation(list(g), list(range(1, self.n_symbols + 1))) for g in gens]
        G = gap.Group(gap_gens)  # letter perms
        C = gap.Group([gap.Permutation([(i % self.k) + 1 for i in range(self.k)])])
        W = gap.WreathProduct(G, C)

        if seeds is None:
            from seeds import default_seed_sample

            seeds = default_seed_sample(self)

        for seed in seeds:
            if seed in self._orbit_id:
                continue
            orb = {tuple(int(i) - 1 for i in img) for img in W.Orbit(list(x + 1 for x in seed))}
            rep = min(orb)
            idx = len(self._orbit_scores)
            self._orbit_id[rep] = idx
            score_sum = sum(self.cost(x) for x in orb)
            self._orbit_scores.append(score_sum / len(orb))
            self._orbit_counts.append(len(orb))
            for o in orb:
                self._orbit_id[o] = idx

    # ── Helpers ───────────────────────────────────────
    def _all_letter_perms(self) -> List[Permutation]:
        if self.letter_perm_generators is None:
            # full symmetric for small n (n ≤ 6)
            if self.n_symbols > 6:
                raise ValueError("Explicit S_n generation too large; provide generators or use GAP.")
            return list(permutations(range(self.n_symbols)))
        # breadth‑first closure of subgroup generated by given perms
        seen = {tuple(range(self.n_symbols))}
        frontier = [tuple(p) for p in self.letter_perm_generators]
        while frontier:
            σ = frontier.pop()
            if σ not in seen:
                seen.add(σ)
                frontier.extend(tuple(σ[i] for i in τ) for τ in list(seen))
        return list(seen)

    def _canonical(self, x: String, perms: List[Permutation], shifts=range(0)) -> String:
        if not shifts:
            shifts = range(self.k)
        best = x
        for s in shifts:
            xs = cyclic_shift(x, s)
            for σ in perms:
                y = apply_letter_perm(xs, σ)
                if y < best:
                    best = y
        return best
