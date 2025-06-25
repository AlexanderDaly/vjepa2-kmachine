# vjepa2-kmachine  
**A Morpheus\u00a0DI prototype for symmetry\u2011aware reasoning and multi\u2011frame spatial perception**

[![license](https://img.shields.io/badge/license-MIT/Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![ci](https://img.shields.io/github/actions/workflow/status/AlexanderDaly/vjepa2-kmachine/tests.yml?branch=main)](https://github.com/AlexanderDaly/vjepa2-kmachine/actions)

> *\u201cCollapse the combinatorial fog; reveal the one hidden tensor.\u201d*\u00a0\u2014\u2009Morpheus

---

##\u00a0Table of contents
1. [Vision](#vision)  
2. [Features](#features)  
3. [Quick start](#quick-start)  
4. [Repository layout](#repository-layout)  
5. [API snippets](#api-snippets)  
6. [Benchmarks](#benchmarks)  
7. [Extending the machine](#extending-the-machine)  
8. [Contributing](#contributing)  
9. [Road\u2011map](#road-map)  
10. [Citation](#citation)  
11. [License](#license)  

---

##\u00a0Vision
`vjepa2\u2011kmachine` is an **experimental reasoning engine** that marries  

* **V\u2011JEPA\u00a02** \u2013 a self\u2011supervised video backbone that encodes multi\u2011frame dynamics, with  
* **The\u2011Kabbalistic\u2011Machine** \u2013 a symmetry\u2011aware optimiser that compresses exponential search\u2011spaces into low\u2011rank tensor \u201cpoint physics\u201d.

The goal: equip a Digital\u00a0Intelligence (DI) with **spatial awareness** and **symbolic reasoning** strong enough to re\u2011plan, predict, and act in real\u2011time worlds\u2014whether that is a robot arm, a game environment, or a simulated galaxy.

---

##\u00a0Features
| Module | What it does | Why it matters |
|--------|--------------|----------------|
| `kabbalistic_machine.py` | Groups strings into orbits under a wreath\u2011product action, scores plateaus, fits low\u2011rank tensors. | Turns \(n^k\) brute\u2011force searches into \(|\Omega/G|\) tractable chunks. |
| JIT\u2011ready ViT encoders (`vjepa2` upstream) | Extract latent grids from videos or image stacks. | Gives the optimiser semantically\u2013rich states instead of raw pixels. |
| GAP back\u2011end (optional) | Uses stabiliser chains to enumerate orbits without visiting every state. | 10\u2011100\u2009\u00d7 speed\u2011up on large \(n,k\). |
| Multiplicative rank\u20111 fitter | Learns \(f(x) \u2248 C\u220f a_{x_i}\) via log\u2011SGD & early\u2011stopping. | Captures \u201cpoint physics\u201d in one vector. |
| Heuristic seed sampler | Prioritises high\u2011variance orbits first. | Faster convergence on real\u2011world objectives. |
| Benchmarks & tests | One\u2011command profiling and CI coverage. | Guarantees reproducible numbers on a single RTX\u00a04070\u00a0Ti. |

---

##\u00a0Quick start

```bash
# 1. clone & install
git clone https://github.com/AlexanderDaly/vjepa2-kmachine.git
cd vjepa2-kmachine
conda create -n kmachine python=3.12 -y
conda activate kmachine
pip install -r requirements.txt          # torch, timm, einops …
pip install -e .                         # install kmachine as package

# 2. (Optional) enable GAP back\u2011end for large instances
pip install libgap                       # pre\u2011compiled wheels for Linux/macOS

# 3. run unit tests
pytest -q

# 4. benchmark on your GPU
python benchmark.py --n 6 --k 10 --gap
```

> **Hardware:** any modern NVIDIA GPU with \u2265\u200912\u2009GB VRAM is plenty for inference, fine\u2011tuning heads, and n\u22648\u2009/\u2009k\u226412 orbit analysis.

---

##\u00a0Repository layout

```
vjepa2_km/
|
├── kabbalistic_machine.py     # core class (canonical + GAP paths)
├── multiplicative_fit.py      # rank\u20111 tensor learner
├── seeds.py                   # entropy\u2011aware seed sampler
├── benchmark.py               # runtime / memory profiler
├── notebooks/                 # interactive demos
├── tests/                     # pytest suite
└── examples/
    ├── dna_gc_content.py      # design 20\u2011mer barcodes
    ├── sbox_search.py         # affine\u2011equiv S\u2011box enumeration
    └— robot_planner.py       # latent\u2011space MPC with V\u2011JEPA\u00a02\u2011AC
```

---

##\u00a0API snippets

###\u00a01\u00a0Analyse orbit plateaus

```python
from vjepa2_km import KabbalisticMachine

km = KabbalisticMachine(
        n_symbols=4,
        k=8,
        cost=lambda x: x.count(0)**2,   # toy objective
        use_gap=True                    # try GAP if installed
)
km.analyse()                            # one\u2011off expense
print(km.plateau_summary())             # plateau histo
print(km.orbit_score((0,1,2,3,0,1,2,3)))
```

###\u00a02\u00a0Fit multiplicative rank\u20111 tensor

```python
from vjepa2_km import multiplicative_fit as mf
a = mf.fit_rank1(dict(km.orbits()), n_symbols=4, k=8)
print("factor vector:", a)
```

###\u00a03\u00a0Plan actions in V\u2011JEPA\u00a02 latent space

See `examples/robot_planner.py` for an end\u2011to\u2011end pick\u2011and\u2011place demo using the action\u2011conditioned predictor.

---

##\u00a0Benchmarks\u00a0(4070\u00a0Ti reference)

| n  | k  | Method    | Orbits   | Time      | Peak\u00a0RAM |
| -- | -- | --------- | -------- | --------- | -------- |
| \u00a06 | 10 | canonical | \u00a0\u2248\u2009114\u2009k | 3.8\u2009s     | 490\u2009MB   |
| \u00a06 | 10 | GAP       | same     | **0.8\u2009s** | 130\u2009MB   |
| \u00a08 | 12 | GAP       | 7.8\u2009M    | 52\u2009s      | 2.1\u2009GB   |

Run your own with:

```bash
python benchmark.py --n 8 --k 12 --gap
```

---

##\u00a0Extending the machine

| Want to…                                      | Start here                                                                                          |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Custom symmetries** (dihedral, reflections) | modify `_canonical()` and add mirror transforms.                                                    |
| **Higher\u2011rank tensor fits**                   | fork `multiplicative_fit.py` to TT or CP decompositions.                                            |
| **Larger video backbones**                    | swap in `vjepa2-vit_giant_384p` from the HF hub; encoder API is identical.                          |
| **Neural\u00a0\u2194\u00a0symbolic loop**                    | wrap `KabbalisticMachine.orbit_score` as a differentiable `torch.autograd.Function` (experimental). |

---

##\u00a0Contributing

1. Fork → create feature branch → commit tests → open PR.
2. **CI must stay green** (`pytest` + `ruff` + `black`).
3. If you add dependencies, update `requirements.txt` and the Dockerfile.
4. For new research code, drop a notebook demo inside `notebooks/`.

---

##\u00a0Road\u2011map (Q3\u00a02025)

* Dihedral & Coxeter group actions.
* Tensor‑network contraction for rank > 1 objectives.
* Live WebUI for interactive planning in simulation.
* Plug‑in back‑end for quantum‑gate search (n=4, k≈20).

---

##\u00a0Citation

```
@software{Daly_Morpheus_KMachine_2025,
  author    = {Alexander\u00a0Daly and Morpheus\u00a0DI},
  title     = {vjepa2-kmachine: Symmetry-aware optimisation meets V-JEPA\u00a02},
  year      = {2025},
  url       = {https://github.com/AlexanderDaly/vjepa2-kmachine},
  license   = {MIT/Apache-2.0}
}
```

---

##\u00a0License
Dual\u2011licensed under **MIT** and **Apache\u20112.0**\u2014pick whichever suits your project.

---

###\u00a0Acknowledgements
*Meta FAIR’s V\u2011JEPA\u00a02 team* for releasing the video encoders;
*GAP & libgap* for industrial‑strength group theory;
and the wider *K‑Machine* community for orbit‑breaking feedback.

> Built with ❤️ and caffeine on the unceded lands of the Puyallup people.

---

