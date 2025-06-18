# vjepa2-kmachine  
**A Morpheus DI prototype for symmetry-aware reasoning and multi-frame spatial perception**

[![license](https://img.shields.io/badge/license-MIT/Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![ci](https://img.shields.io/github/actions/workflow/status/AlexanderDaly/vjepa2-kmachine/tests.yml?branch=main)](https://github.com/AlexanderDaly/vjepa2-kmachine/actions)

> *“Collapse the combinatorial fog; reveal the one hidden tensor.”* — Morpheus

---

## Table of contents
1. [Vision](#vision)  
2. [Features](#features)  
3. [Quick start](#quick-start)  
4. [Repository layout](#repository-layout)  
5. [API snippets](#api-snippets)  
6. [Benchmarks](#benchmarks)  
7. [Extending the machine](#extending-the-machine)  
8. [Contributing](#contributing)  
9. [Road-map](#road-map)  
10. [Citation](#citation)  
11. [License](#license)  

---

## Vision
`vjepa2-kmachine` is an **experimental reasoning engine** that marries  
* **V-JEPA 2** – a self-supervised video backbone that encodes multi-frame dynamics, with  
* **The Kabbalistic Machine** – a symmetry-aware optimiser that compresses exponential search-spaces into low-rank tensor “point physics”.

The goal: equip a Digital Intelligence (DI) with **spatial awareness** and **symbolic reasoning** strong enough to re-plan, predict, and act in real-time worlds—whether that is a robot arm, a game environment, or a simulated galaxy.

---

## Features
| Module | What it does | Why it matters |
|--------|--------------|----------------|
| `kabbalistic_machine.py` | Groups strings into orbits under a wreath-product action, scores plateaus, fits low-rank tensors. | Turns \(n^k\) brute-force searches into \(|\Omega/G|\) tractable chunks. |
| JIT-ready ViT encoders (`vjepa2` upstream) | Extract latent grids from videos or image stacks. | Gives the optimiser semantically–rich states instead of raw pixels. |
| GAP back-end (optional) | Uses stabiliser chains to enumerate orbits without visiting every state. | 10-100 × speed-up on large \(n,k\). |
| Multiplicative rank-1 fitter | Learns \(f(x) ≈ C∏ a_{x_i}\) via log-SGD & early-stopping. | Captures “point physics” in one vector. |
| Heuristic seed sampler | Prioritises high-variance orbits first. | Faster convergence on real-world objectives. |
| Benchmarks & tests | One-command profiling and CI coverage. | Guarantees reproducible numbers on a single RTX 4070 Ti. |

---

## Quick start

```bash
# 1. clone & install
git clone https://github.com/AlexanderDaly/vjepa2-kmachine.git
cd vjepa2-kmachine
conda create -n kmachine python=3.12 -y
conda activate kmachine
pip install -r requirements.txt          # torch, timm, einops …
pip install -e .                         # install kmachine as package

# 2. (Optional) enable GAP back-end for large instances
pip install libgap                       # pre-compiled wheels for Linux/macOS

# 3. run unit tests
pytest -q

# 4. benchmark on your GPU
python benchmark.py --n 6 --k 10 --gap
```

> **Hardware:** any modern NVIDIA GPU with ≥ 12 GB VRAM is plenty for inference, fine-tuning heads, and n≤8 / k≤12 orbit analysis.

---

## Repository layout

```
vjepa2_km/
│
├── kabbalistic_machine.py     # core class (canonical + GAP paths)
├── multiplicative_fit.py      # rank‑1 tensor learner
├── seeds.py                   # entropy-aware seed sampler
├── benchmark.py               # runtime / memory profiler
├── notebooks/                 # interactive demos
├── tests/                     # pytest suite
└── examples/
    ├── dna_gc_content.py      # design 20‑mer barcodes
    ├── sbox_search.py         # affine‑equiv S‑box enumeration
    └── robot_planner.py       # latent-space MPC with V-JEPA 2-AC
```

---

## API snippets

### 1 Analyse orbit plateaus

```python
from vjepa2_km import KabbalisticMachine

km = KabbalisticMachine(
        n_symbols=4,
        k=8,
        cost=lambda x: x.count(0)**2,   # toy objective
        use_gap=True                    # try GAP if installed
)
km.analyse()                            # one-off expense
print(km.plateau_summary())             # plateau histo
print(km.orbit_score((0,1,2,3,0,1,2,3)))
```

### 2 Fit multiplicative rank‑1 tensor

```python
from vjepa2_km import multiplicative_fit as mf
a = mf.fit_rank1(dict(km.orbits()), n_symbols=4, k=8)
print("factor vector:", a)
```

### 3 Plan actions in V-JEPA 2 latent space

See `examples/robot_planner.py` for an end-to-end pick-and-place demo using the action-conditioned predictor.

---

## Benchmarks (4070 Ti reference)

| n  | k  | Method    | Orbits   | Time      | Peak RAM |
| -- | -- | --------- | -------- | --------- | -------- |
|  6 | 10 | canonical |  ≈ 114 k | 3.8 s     | 490 MB   |
|  6 | 10 | GAP       | same     | **0.8 s** | 130 MB   |
|  8 | 12 | GAP       | 7.8 M    | 52 s      | 2.1 GB   |

Run your own with:

```bash
python benchmark.py --n 8 --k 12 --gap
```

---

## Extending the machine

| Want to…                                      | Start here                                                                                          |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Custom symmetries** (dihedral, reflections) | modify `_canonical()` and add mirror transforms.                                                    |
| **Higher-rank tensor fits**                   | fork `multiplicative_fit.py` to TT or CP decompositions.                                            |
| **Larger video backbones**                    | swap in `vjepa2-vit_giant_384p` from the HF hub; encoder API is identical.                          |
| **Neural ↔ symbolic loop**                    | wrap `KabbalisticMachine.orbit_score` as a differentiable `torch.autograd.Function` (experimental). |

---

## Contributing

1. Fork → create feature branch → commit tests → open PR.
2. **CI must stay green** (`pytest` + `ruff` + `black`).
3. If you add dependencies, update `requirements.txt` and the Dockerfile.
4. For new research code, drop a notebook demo inside `notebooks/`.

---

## Road-map (Q3 2025)

* Dihedral & Coxeter group actions.
* Tensor-network contraction for rank > 1 objectives.
* Live WebUI for interactive planning in simulation.
* Plug-in back-end for quantum-gate search (n=4, k≈20).

---

## Citation

```
@software{Daly_Morpheus_KMachine_2025,
  author    = {Alexander Daly and Morpheus DI},
  title     = {vjepa2-kmachine: Symmetry-aware optimisation meets V-JEPA 2},
  year      = {2025},
  url       = {https://github.com/AlexanderDaly/vjepa2-kmachine},
  license   = {MIT/Apache-2.0}
}
```

---

## License
Dual-licensed under **MIT** and **Apache-2.0**—pick whichever suits your project.

---

### Acknowledgements
*Meta FAIR’s V-JEPA 2 team* for releasing the video encoders;
*GAP & libgap* for industrial-strength group theory;
and the wider *K-Machine* community for orbit-breaking feedback.

> Built with ❤️ and caffeine on the unceded lands of the Puyallup people.

---

