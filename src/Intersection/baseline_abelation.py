#!/usr/bin/env python3
"""
baseline_abelation (seeded by complete_abelation SAFE set)
==========================================================

Baseline to compare against NPACE-Influence: BOTH agents are modeled as
noisy-rational humans that do Q-MDP over the other agent's intent.

- Agent 0 ("human"): IntersectNoisyRationalHuman(stochastic=True)
- Agent 1 ("robot"): IntersectNoisyRationalHuman(stochastic=False, ctrl_index_self=1)

Loads the SAFE seeds from 'complete_abelation_results.pkl' (produced by the
complete-info ablation) and runs the baseline on those initial conditions.

Artifacts:
  • baseline_abelation_results.pkl
  • baseline_abelation_xy.png
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import pickle
import pathlib
from typing import Dict, Any, List

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Append the root directory so we can import local modules
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent  # src/
sys.path.append(str(ROOT_DIR))

from iLQGame.multiplayer_dynamical_system import UncontrolledIntersection2PlayerSystem
from intersection_noisy_rational_human import IntersectNoisyRationalHuman
from Intersection_blame_me_final import IntersectBlameMeController

# ---------------------------------------------------------------------------
# Experiment parameters (identical to influence ablation)
# ---------------------------------------------------------------------------
# Geometry and smoothing constants
R: float = 70.0
W: float = 1.5
L: float = 3.0

# Soft window parameters
GAMMA: float = 1.0         # logistic sharpness for σ
B_PEN: float = 1e6       # collision penalty weight
MU: float = 1e-6         # terminal progress weight
V_NOM: float = 16.0      # desired terminal speed

# Simulation timing
DT: float = 0.1
HORIZON: int = 20        # 3 s horizon (H=30 at dt=0.1)
SIM_STEPS: int = 30      # 3 s rollout

# Control limits (wide bounds)
ACC_LOW: float = -10.0
ACC_HIGH: float =  10.0

# Intent set for both agents (kept for controller banks)
INTENT_SET = [1, 3, 5]

# Crossing centre for plots (relative coordinate)
D_CROSS: float = R / 2.0  # 35.0 m when R=70

# Input/Output filenames
SAFE_SEEDS_PKL = SCRIPT_DIR / "complete_abelation_results.pkl"   # from complete-info ablation
RESULTS_PKL    = SCRIPT_DIR / "baseline_abelation_results.pkl"
XY_PLOT_PNG    = SCRIPT_DIR / "baseline_abelation_xy.png"


# --- helpers for reusing controller banks (identical pattern) ---
def _reset_agent_controller_state(agent: IntersectNoisyRationalHuman) -> None:
    """Reset agent's belief and cache; keep solver warm-starts."""
    nOpp = len(agent._intents)
    if isinstance(agent._belief, np.ndarray):
        agent._belief[:] = 1.0 / nOpp
    else:
        agent._belief = np.ones(nOpp, dtype=np.float64) / nOpp
    agent._state_cache = None
    # refresh seed if agent is stochastic (keeps trials independent)
    try:
        if getattr(agent, "_stochastic", False):
            import random
            agent.set_seed(random.getrandbits(32))
    except Exception:
        pass


def run_trial(human: IntersectNoisyRationalHuman,
              robot: IntersectBlameMeController,
              theta_h: int,
              theta_r: int,
              init_state: jnp.ndarray) -> Dict[str, Any]:
    """Run a single simulation trial and return logs (controllers reused)."""

    # Dynamics
    sys_dyn = UncontrolledIntersection2PlayerSystem(T=DT)

    # logs
    states: List[np.ndarray] = []
    a1_list: List[float] = []
    a2_list: List[float] = []
    belief_h: List[float] = []  # human's belief about the (robot) true theta_r
    belief_r: List[float] = []  # robot's belief about the (human) true theta_h

    # Last observed accelerations for Bayesian update (not used by the model, but passed)
    a1_prev, a2_prev = 0.00, 0.00
    x_curr = init_state

    for _ in range(SIM_STEPS):
        # Compute controls; each uses last observed opponent acceleration (API-compatible)
        a1_cmd = human.compute_action(obs=np.asarray(x_curr), a_opponent_observed=a2_prev)
        a2_cmd = robot.compute_action(obs=np.asarray(x_curr), a_opponent_observed=a1_prev)

        # Log current state and control
        states.append(np.asarray(x_curr, dtype=float))
        a1_list.append(float(a1_cmd))
        a2_list.append(float(a2_cmd))

        # Beliefs: probability mass on the ground-truth intent of the other
        b_h = getattr(human, "belief_over_theta", {})
        b_r = getattr(robot, "belief_over_theta", {})
        try:
            belief_h.append(float(b_h.get(theta_r, 0.0)))
        except Exception:
            belief_h.append(0.0)
        try:
            belief_r.append(float(b_r.get(theta_h, 0.0)))
        except Exception:
            belief_r.append(0.0)

        # Plant step
        x_curr = sys_dyn.disc_time_dyn(x_curr, [jnp.array([a1_cmd]), jnp.array([a2_cmd])])

        # update observed accelerations
        a1_prev, a2_prev = a1_cmd, a2_cmd

    return {
        "theta_h": theta_h,
        "theta_r": theta_r,
        "states": np.stack(states, axis=1),
        "a1": np.asarray(a1_list, dtype=float),
        "a2": np.asarray(a2_list, dtype=float),
        "belief_h": np.asarray(belief_h, dtype=float),
        "belief_r": np.asarray(belief_r, dtype=float)
    }


def main() -> None:
    """Run baseline (both agents noisy-rational) on SAFE seeds and save results."""

    # Load SAFE seeds (list of dicts) from complete-info ablation
    if not SAFE_SEEDS_PKL.exists():
        raise FileNotFoundError(f"Required seed file not found: {SAFE_SEEDS_PKL}")
    with open(SAFE_SEEDS_PKL, "rb") as f:
        safe_seeds = pickle.load(f)

    if not isinstance(safe_seeds, list) or len(safe_seeds) == 0:
        raise ValueError("Loaded seeds are empty or not a list.")

    # Number of trials is the number of SAFE seeds
    N_TRIALS = len(safe_seeds)

    # --- build controller banks once and reuse across trials ---
    # Agent 0 (human): control index 0, stochastic
    human_bank = {
        th: IntersectNoisyRationalHuman(
            theta_self=th,
            ctrl_index_self=0,
            intents=tuple(INTENT_SET),
            dt=DT, horizon=HORIZON,
            effort_w=1.0, b_pen=B_PEN, gamma=GAMMA,
            mu=MU, v_nom=V_NOM, R=R, W=W, L=L,
            acc_low=ACC_LOW, acc_high=ACC_HIGH,
            beta=1.0, stochastic=True,   # <-- stochastic human
            verbose=False
        )
        for th in INTENT_SET
    }

    # Agent 1 (robot side in dynamics): same model but deterministic & ctrl_index=1
    robot_bank = {
        tr: IntersectBlameMeController(
            theta_self=tr,
            ctrl_index_self=1,          # <-- this controls player-1 a2
            intents=tuple(INTENT_SET),  # beliefs over the other agent's intents
            dt=DT, horizon=HORIZON,
            effort_w=1.0, b_pen=B_PEN, gamma=GAMMA,
            mu=MU, v_nom=V_NOM, R=R, W=W, L=L,
            acc_low=ACC_LOW, acc_high=ACC_HIGH,
            sigma2_action_obs=1.1,
            beta=1.0, stochastic=False, # <-- deterministic “robot” baseline
            verbose=False
        )
        for tr in INTENT_SET
    }

    results: List[Dict[str, Any]] = []

    for idx, seed in enumerate(safe_seeds, start=1):
        # Extract θ and init_state from the SAFE seed
        theta_h = int(seed["theta_h"])
        theta_r = int(seed["theta_r"])
        init_state = jnp.array(seed["init_state"], dtype=jnp.float32)

        # Grab controllers from the banks and reset their beliefs/caches
        human = human_bank[theta_h]
        robot = robot_bank[theta_r]
        _reset_agent_controller_state(human)
        _reset_agent_controller_state(robot)

        # Run one trial with the seeded ICs (no safety check here)
        res = run_trial(human, robot, theta_h, theta_r, init_state)
        results.append(res)

        # Print progress (same style)
        print(f"Trial {idx}/{N_TRIALS} complete: θ_h={theta_h}, θ_r={theta_r}")

    # Save results to pickle
    with open(RESULTS_PKL, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {RESULTS_PKL}")

    # Plot aggregated XY trajectories with fixed view and collision square
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(8, 6))

    # Add 3×3 gray square centered at (0,0) as collision area (same as before)
    ax.add_patch(Rectangle((-2.25, -2.25), 4.5, 4.5,
                           facecolor="#bbbbbb", edgecolor="#888888",
                           linewidth=1.0, alpha=0.5, zorder=0))

    for res in results:
        xs = res["states"]
        # x = d2 - D_CROSS (horizontal car), y = d1 - D_CROSS (vertical car)
        x_xy = xs[2] - D_CROSS
        y_xy = xs[0] - D_CROSS
        ax.plot(x_xy, y_xy, linewidth=1.0, alpha=0.7, zorder=2)

    ax.axhline(0, color="#cccccc", lw=1, zorder=1)
    ax.axvline(0, color="#cccccc", lw=1, zorder=1)
    ax.set_xlabel("x = d2 − D_CROSS (m)")
    ax.set_ylabel("y = d1 − D_CROSS (m)")
    ax.set_title(f"Aggregated XY trajectories ({N_TRIALS} trials) — Baseline (both noisy-rational)")
    ax.grid(True, alpha=0.3)

    # Fix view to [-15, 15] on both axes and keep equal scaling
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)

    plt.tight_layout()
    plt.savefig(XY_PLOT_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved aggregated XY plot to {XY_PLOT_PNG}")


if __name__ == "__main__":
    main()
