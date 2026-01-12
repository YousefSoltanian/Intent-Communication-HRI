# Intent-Communication-HRI

This repository contains my IJRR research codebase for **intent communication in human–robot interaction (HRI)**.

At a high level, the robot plans in a game-theoretic setting while explicitly modeling the human as an **uncertain (learning) agent**. The human is modeled with **ILQGame/QMDP-style decision-making**, and intent inference is performed via **Bayesian belief updates**.

## Included examples
- **Intersection driving** (two-agent interaction / intent inference)
- **Social navigation** (planar navigation in a corridor/hallway)
- **HRI Lunar Lander shared control** (human + robot shared-control experiments)

Most components are implemented in **JAX** (and “jaxified”) for efficient simulation and real-time-ish performance.

The `src/HRI/` folder includes experiment runners for a **human interacting with the Lunar Lander** on a computer.

Across scenarios, the primary robot controller is **N-PACE Influence**. Simulation runners / experiment runners typically include **`demo`** in the filename.

---

## Quickstart (tested on Ubuntu)

```bash
# 1) Clone
git clone https://github.com/YousefSoltanian/Intent-Communication-HRI.git
cd Intent-Communication-HRI

# 2) Create the conda env (one-time)
conda env create -f environment.yml

# 3) Run the navigation demo (no activation required)
MPLBACKEND=Agg PYTHONPATH="$(pwd)/src" \
conda run -n intent_comm_nav python src/Navigation/navigation_npace_influence_vs_human_demo.py
```

---

## Compatibility notes
- **NumPy is pinned to `1.26.4`**  
  NumPy 2.x can break older compiled wheels used by common scientific packages.
- **JAX/JAXLIB are pinned to `0.4.23`**  
  This repo relies on compatibility with internal imports used by `iLQGame`.
- **Headless plotting**  
  Demo scripts run headless via `MPLBACKEND=Agg` to avoid GUI backend differences.

---

## Repository structure (high level)
- `src/` – main source code (controllers, dynamics, solvers, demos)
- `src/iLQGame/` – iLQGame-based solver and supporting code
- `src/Navigation/` – social navigation scenario + demos
- `src/Intersection/` – intersection scenario + demos
- `src/HRI/` – lunar lander shared control + experiment runners
- `scripts/` – runnable entrypoints (demo scripts, environment helpers)

---

## Running demos (no `.sh` required)

All demos can be run directly with `python` via `conda run` (recommended: avoids pyenv/python2 PATH issues).

```bash
# From the repo root:
MPLBACKEND=Agg PYTHONPATH="$(pwd)/src" \
conda run -n intent_comm_nav python <path/to/demo_script.py>
```

Examples:
```bash
MPLBACKEND=Agg PYTHONPATH="$(pwd)/src" \
conda run -n intent_comm_nav python src/Navigation/navigation_npace_influence_vs_human_demo.py
```

---

## Citation

If you use this code in academic work, please cite the associated IJRR submission (details to be added).
