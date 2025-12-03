#!/usr/bin/env python3
"""
complete_abelation
==================

Complete-information ablation for the uncontrolled intersection ILQ game.
Same geometry, timing, sampling, and plotting frame as your NPACE ablation,
but here both players know the intents (complete information). We run 600
trials and keep ONLY SAFE interactions.

Safety rule (per spec):
  A trajectory is UNSAFE if there exists any time step k where
    |d1[k] - R/2| < 3.0  AND  |d2[k] - R/2| < 3.0
  Additionally, to catch crossings between samples, a trajectory is UNSAFE
  if ANY line segment between successive samples intersects the box
  [-3, 3] × [-3, 3] in the XY frame (x = d2 - R/2, y = d1 - R/2).

Artifacts saved beside this file:
  • complete_abelation_results.pkl     – list of dicts for SAFE trials only
  • complete_abelation_xy_safe.png     – XY plot of SAFE trajectories only
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import random
import pickle
import pathlib
from typing import Dict, Any, List, Tuple

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# ───────────────────────── paths / imports ─────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent  # src/
sys.path.append(str(ROOT_DIR))

from iLQGame.multiplayer_dynamical_system import UncontrolledIntersection2PlayerSystem
from iLQGame.cost        import UncontrolledIntersectionPlayerCost
from iLQGame.player_cost import PlayerCost
from iLQGame.constraint  import BoxConstraint
from iLQGame.ilq_solver  import ILQSolver

# ---------------------------------------------------------------------------
# Experiment parameters (mirror your latest NPACE ablation)
# ---------------------------------------------------------------------------
# Geometry and smoothing constants
R: float = 70.0
W: float = 1.5
L: float = 3.0

# Soft window parameters / costs
GAMMA: float = 1.0        # logistic sharpness for σ
B_PEN: float = 1e5      # collision penalty weight
MU: float = 1e-6       # terminal progress weight (10e-0)
V_NOM: float = 16.0     # desired terminal speed

# Simulation timing
DT: float = 0.1
HORIZON: int = 10       # 3 s horizon (H=30 at dt=0.1)
SIM_STEPS: int = 30     # 3 s rollout

# Control limits
ACC_LOW: float = -10.0
ACC_HIGH: float =  10.0

# Intent set for both agents (complete info picks one per player)
INTENT_SET = [1, 3, 5]

# Initial state sampling ranges
POS_RANGE = (12.0, 16.0)  # [min, max] for d1, d2
VEL_RANGE = (14.0, 17.0)  # [min, max] for v1, v2

# Number of trials (complete-info ablation)
N_TRIALS: int = 500

# Crossing centre for plots (relative coordinate)
D_CROSS: float = R / 2.0  # = 35.0 m

# Output filenames
RESULTS_PKL = SCRIPT_DIR / "complete_abelation_results.pkl"
XY_PLOT_PNG = SCRIPT_DIR / "complete_abelation_xy_safe.png"

# Safety box half-size in meters (your rule uses ±3.0 m)
HALF_BOX = 2.25


def _build_solver(theta1: int, theta2: int) -> Tuple[ILQSolver, UncontrolledIntersection2PlayerSystem]:
    """Construct an ILQ solver for given complete-information intents and return (solver, dynamics)."""
    sys_dyn = UncontrolledIntersection2PlayerSystem(T=DT)

    pc_1 = PlayerCost()
    pc_1.add_cost(
        UncontrolledIntersectionPlayerCost(
            player_index=0, theta_self=float(theta1), horizon=HORIZON,
            effort_weight=1.0, b=B_PEN, gamma=GAMMA,
            mu=MU, v_nom=V_NOM, R=R, W=W, L=L
        ),
        arg="x", weight=1.0
    )

    pc_2 = PlayerCost()
    pc_2.add_cost(
        UncontrolledIntersectionPlayerCost(
            player_index=1, theta_self=float(theta2), horizon=HORIZON,
            effort_weight=1.0, b=B_PEN, gamma=GAMMA,
            mu=MU, v_nom=V_NOM, R=R, W=W, L=L
        ),
        arg="x", weight=1.0
    )

    acc_limit_1 = BoxConstraint(ACC_LOW, ACC_HIGH)
    acc_limit_2 = BoxConstraint(ACC_LOW, ACC_HIGH)

    Ps0 = jnp.zeros((1, 4, HORIZON))
    al0 = jnp.zeros((1,      HORIZON))

    solver = ILQSolver(
        dynamics      = sys_dyn,
        player_costs  = [pc_1, pc_2],
        Ps            = [Ps0, Ps0],
        alphas        = [al0, al0],
        max_iter      = 25,
        u_constraints = [acc_limit_1, acc_limit_2],
        verbose       = False
    )
    return solver, sys_dyn


def _run_trial(theta_h: int, theta_r: int, init_state: jnp.ndarray) -> Dict[str, Any]:
    """Run one complete-information ILQ rollout and return logs."""
    solver, sys_dyn = _build_solver(theta_h, theta_r)

    states: List[np.ndarray] = []
    a1_list: List[float] = []
    a2_list: List[float] = []

    x_curr = init_state
    for _ in range(SIM_STEPS):
        solver.run(x_curr)

        # First control at this step for each player (match your accessor)
        a1 = float(solver._best_operating_point[1][0][0, 0])
        a2 = float(solver._best_operating_point[1][1][0][0])

        states.append(np.asarray(x_curr, dtype=float))
        a1_list.append(a1)
        a2_list.append(a2)

        x_curr = sys_dyn.disc_time_dyn(x_curr, [jnp.array([a1]), jnp.array([a2])])

    return {
        "theta_h": int(theta_h),
        "theta_r": int(theta_r),
        "init_state": np.asarray(init_state, float),   # include sampled initial state
        "states": np.stack(states, axis=1),           # (4, T): [d1, v1, d2, v2]
        "a1": np.asarray(a1_list, dtype=float),       # (T,)
        "a2": np.asarray(a2_list, dtype=float),       # (T,)
        # Structural parity with NPACE logs:
        "belief_h": np.full(SIM_STEPS, np.nan),
        "belief_r": np.full(SIM_STEPS, np.nan),
    }


def _segment_intersects_box(x0: float, y0: float, x1: float, y1: float,
                            hx: float, hy: float) -> bool:
    """
    Liang–Barsky style 'slab' test for segment-box intersection.
    Box is axis-aligned: [-hx, +hx] × [-hy, +hy].
    """
    dx = x1 - x0
    dy = y1 - y0

    # Helper to compute t-interval where axis is inside its slab
    def interval_1d(p0, dp, h):
        if abs(dp) < 1e-12:
            # Parallel to axis: inside slab for all t iff |p0| <= h
            return (0.0, 1.0) if abs(p0) <= h else (1.0, 0.0)  # empty if swapped
        t0 = (-h - p0) / dp
        t1 = ( h - p0) / dp
        if t0 > t1:
            t0, t1 = t1, t0
        # clamp to [0,1]
        return (max(0.0, t0), min(1.0, t1))

    tx0, tx1 = interval_1d(x0, dx, hx)
    ty0, ty1 = interval_1d(y0, dy, hy)

    # Intersection interval is the overlap of x and y intervals
    t_enter = max(tx0, ty0)
    t_exit  = min(tx1, ty1)
    return t_enter <= t_exit


def _is_safe(traj_states: np.ndarray) -> bool:
    """
    SAFE iff for all k we DO NOT have (|d1-R/2| < 3 AND |d2-R/2| < 3),
    and no segment between successive samples intersects the central box.
    """
    d1 = traj_states[0]  # shape (T,)
    d2 = traj_states[2]  # shape (T,)

    # Vertex test (sampled points)
    cond1 = np.abs(d1 - D_CROSS) < HALF_BOX
    cond2 = np.abs(d2 - D_CROSS) < HALF_BOX
    if np.any(cond1 & cond2):
        return False

    # Segment test (between successive samples) in XY frame
    x = d2 - D_CROSS
    y = d1 - D_CROSS
    for k in range(len(x) - 1):
        if _segment_intersects_box(x[k], y[k], x[k+1], y[k+1], HALF_BOX, HALF_BOX):
            return False

    return True


def main() -> None:
    safe_results: List[Dict[str, Any]] = []
    safe_count = 0

    for trial in range(1, N_TRIALS + 1):
        # Sample complete-information intents (uniform)
        theta_h = random.choice(INTENT_SET)
        theta_r = random.choice(INTENT_SET)

        # Sample initial conditions
        d1 = random.uniform(*POS_RANGE)
        v1 = random.uniform(*VEL_RANGE)
        d2 = random.uniform(*POS_RANGE)
        v2 = random.uniform(*VEL_RANGE)
        init_state = jnp.array([d1, v1, d2, v2], dtype=jnp.float32)

        # Rollout
        res = _run_trial(theta_h, theta_r, init_state)

        # Safety filter (per spec + segment crossing)
        if _is_safe(res["states"]):
            safe_results.append(res)
            safe_count += 1

        # Progress (keep your style)
        print(f"Trial {trial}/{N_TRIALS} complete: θ_h={theta_h}, θ_r={theta_r}")

    # Save only SAFE interactions
    with open(RESULTS_PKL, "wb") as f:
        pickle.dump(safe_results, f)
    print(f"Saved SAFE results to {RESULTS_PKL}")

    # Report safety percentage
    pct_safe = 100.0 * safe_count / float(N_TRIALS)
    print(f"Safe interactions: {safe_count}/{N_TRIALS} ({pct_safe:.2f}%)")

    # Plot only SAFE trajectories with the central ±3 m square
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(8, 6))

    # Central 6×6 m square (±3 m) at (0,0) in the XY plot frame
    ax.add_patch(Rectangle((-HALF_BOX, -HALF_BOX), 2*HALF_BOX, 2*HALF_BOX,
                           facecolor="#bbbbbb", edgecolor="#888888",
                           linewidth=1.0, alpha=0.5, zorder=0))

    for res in safe_results:
        xs = res["states"]
        x_xy = xs[2] - D_CROSS  # horizontal car (d2 - crossing)
        y_xy = xs[0] - D_CROSS  # vertical car   (d1 - crossing)
        ax.plot(x_xy, y_xy, linewidth=1.0, alpha=0.8, zorder=2)

    ax.axhline(0, color="#cccccc", lw=1, zorder=1)
    ax.axvline(0, color="#cccccc", lw=1, zorder=1)
    ax.set_xlabel("x = d2 − D_CROSS (m)")
    ax.set_ylabel("y = d1 − D_CROSS (m)")
    ax.set_title(f"SAFE XY trajectories ({safe_count}/{N_TRIALS}; {pct_safe:.2f}%)")
    ax.grid(True, alpha=0.3)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)

    plt.tight_layout()
    plt.savefig(XY_PLOT_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved SAFE XY plot to {XY_PLOT_PNG}")


if __name__ == "__main__":
    main()
