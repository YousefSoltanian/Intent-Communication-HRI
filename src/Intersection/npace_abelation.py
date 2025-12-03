#!/usr/bin/env python3
"""
npace_abelation (seeded by complete_abelation SAFE set)
=======================================================

Same NPACE vs. human ablation as before (geometry, timing, logging, plotting),
but instead of randomly sampling (θ_h, θ_r, d1,v1,d2,v2), we load the SAFE
seeds from 'complete_abelation_results.pkl' (produced by the complete-info
ablation) and run NPACE on those initial conditions/intents.

Artifacts (same names/format as original NPACE ablation):
  • npace_abelation_results.pkl
  • npace_abelation_xy.png
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
from intersection_npace import IntersectionNPACE

# ---------------------------------------------------------------------------
# Experiment parameters (unchanged from your NPACE ablation)
# ---------------------------------------------------------------------------
# Geometry and smoothing constants
R: float = 70.0
W: float = 1.5
L: float = 3.0

# Soft window parameters
GAMMA: float = 1        # logistic sharpness for σ
B_PEN: float = 1e5      # collision penalty weight
MU: float = 10e-0       # terminal progress weight
V_NOM: float = 18.0     # desired terminal speed

# Simulation timing
DT: float = 0.1
HORIZON: int = 30       # 3 s horizon (H=30 at dt=0.1)
SIM_STEPS: int = 30     # 3 s rollout

# Control limits (wide bounds)
ACC_LOW: float = -50.0
ACC_HIGH: float =  50.0

# Intent set for both agents (kept for controller banks)
INTENT_SET = [1, 2, 3, 4, 5]

# Crossing centre for plots (relative coordinate)
D_CROSS: float = R / 2.0  # 35.0 m when R=70

# Input/Output filenames
SAFE_SEEDS_PKL = SCRIPT_DIR / "complete_abelation_results.pkl"   # from complete-info ablation
RESULTS_PKL    = SCRIPT_DIR / "npace_abelation_results.pkl"
XY_PLOT_PNG    = SCRIPT_DIR / "npace_abelation_xy.png"


# --- helpers for reusing controller banks (unchanged logic) ---
def _reset_robot_controller_state(robot: IntersectionNPACE) -> None:
    """Reset only beliefs and caches; keep solver warm-starts."""
    nH = len(robot._intents)
    nR = nH
    # robot belief over human θ_h
    if isinstance(robot._b_h, np.ndarray):
        robot._b_h[:] = 1.0 / nH
    else:
        robot._b_h = np.ones(nH, dtype=np.float64) / nH
    # modeled human beliefs over robot θ_r per θ_h
    if isinstance(robot._q_r, dict):
        for ih in range(nH):
            robot._q_r[ih][:] = 1.0 / nR
    else:
        robot._q_r[:] = 1.0 / nR
    # caches
    robot._pred_ctrl = None
    robot._x_cache = None


def _reset_human_controller_state(human: IntersectNoisyRationalHuman) -> None:
    """Reset human belief and cache; keep its solver warm-starts."""
    nR = len(human._intents)
    if isinstance(human._belief, np.ndarray):
        human._belief[:] = 1.0 / nR
    else:
        human._belief = np.ones(nR, dtype=np.float64) / nR
    human._state_cache = None
    # preserve stochasticity while emulating "fresh controller" randomness:
    try:
        import random
        human.set_seed(random.getrandbits(32))
    except Exception:
        pass


def run_trial(human: IntersectNoisyRationalHuman,
              robot: IntersectionNPACE,
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
    belief_h: List[float] = []  # human's belief about robot true theta
    belief_r: List[float] = []  # robot's belief about human true theta

    # Last observed accelerations for Bayesian update
    a1_prev, a2_prev = 0.001, 0.001
    x_curr = init_state

    for _ in range(SIM_STEPS):
        # Compute controls; each uses last observed opponent acceleration
        a1_cmd = human.compute_action(obs=np.asarray(x_curr), a_opponent_observed=a2_prev)
        a2_cmd = robot.compute_action(obs=np.asarray(x_curr), a1_observed=a1_prev)

        # Log current state and control
        states.append(np.asarray(x_curr, dtype=float))
        a1_list.append(float(a1_cmd))
        a2_list.append(float(a2_cmd))

        # Beliefs: probability assigned to the true intent of the other
        b_h = getattr(human, "belief_over_theta", {})
        b_r = getattr(robot, "robot_belief_over_human", {})
        # b_h / b_r may be dict-like; be defensive:
        try:
            belief_h.append(float(b_h.get(theta_r, 0.0)))
        except Exception:
            belief_h.append(float(0.0))
        try:
            belief_r.append(float(b_r.get(theta_h, 0.0)))
        except Exception:
            belief_r.append(float(0.0))

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
    """Run NPACE on the SAFE seeds from complete_abelation and save results."""

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
    human_bank = {
        th: IntersectNoisyRationalHuman(
            theta_self=th,
            ctrl_index_self=0,
            dt=DT, horizon=HORIZON,
            effort_w=1.0, b_pen=B_PEN, gamma=GAMMA,
            mu=MU, v_nom=V_NOM, R=R, W=W, L=L,
            acc_low=ACC_LOW, acc_high=ACC_HIGH,
            beta=1.0, stochastic=True
        )
        for th in INTENT_SET
    }

    robot_bank = {
        tr: IntersectionNPACE(
            theta_robot_true=tr,
            dt=DT, horizon=HORIZON,
            effort_w=1.0, b_pen=B_PEN, gamma=GAMMA,
            mu=MU, v_nom=V_NOM, R=R, W=W, L=L,
            acc_low=ACC_LOW, acc_high=ACC_HIGH,
            beta_state=1.0, rho_forget=0.0,
            sigma2_state=(25.0**2, 4.0**2, 25.0**2, 4.0**2),
            max_iter=25, verbose=False
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
        _reset_human_controller_state(human)
        _reset_robot_controller_state(robot)

        # Run one trial with the seeded ICs (no safety check here)
        res = run_trial(human, robot, theta_h, theta_r, init_state)
        results.append(res)

        # Print progress (same style)
        print(f"Trial {idx}/{N_TRIALS} complete: θ_h={theta_h}, θ_r={theta_r}")

    # Save results to pickle (same filename/structure as original NPACE ablation)
    with open(RESULTS_PKL, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {RESULTS_PKL}")

    # Plot aggregated XY trajectories with fixed view and collision square
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(8, 6))

    # Add 3×3 gray square centered at (0,0) as collision area (same as before)
    ax.add_patch(Rectangle((-1.5, -1.5), 3.0, 3.0,
                           facecolor="#bbbbbb", edgecolor="#888888",
                           linewidth=1.0, alpha=0.5, zorder=0))

    for res in results:
        xs = res["states"]
        # x coordinate = d2 - D_CROSS (horizontal car), y coordinate = d1 - D_CROSS (vertical car)
        x_xy = xs[2] - D_CROSS
        y_xy = xs[0] - D_CROSS
        ax.plot(x_xy, y_xy, linewidth=1.0, alpha=0.7, zorder=2)

    ax.axhline(0, color="#cccccc", lw=1, zorder=1)
    ax.axvline(0, color="#cccccc", lw=1, zorder=1)
    ax.set_xlabel("x = d2 − D_CROSS (m)")
    ax.set_ylabel("y = d1 − D_CROSS (m)")
    ax.set_title(f"Aggregated XY trajectories ({N_TRIALS} trials)")
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
