#!/usr/bin/env python
"""
intersection_npace_influence_vs_human_demo.py
============================================

This script runs a two‑player simulation on the uncontrolled intersection
with one agent using the NPACE‑Influence controller and the other using a
noisy‑rational human model. The setup matches your earlier NPACE vs Human
example exactly: same time step, horizon length, geometry, penalty scales,
and control limits.  The only difference is that the robot (vehicle‑2) now
uses the NPACE‑Influence controller, which adds a teaching term to its
Q‑MDP objective to accelerate the human’s understanding of the robot’s true
intent.  The human (vehicle‑1) still follows a Q‑MDP policy with β=1.

Artifacts generated:

  - intersection_npace_influence_vs_human_xy.png      (trajectory plot)
  - intersection_npace_influence_vs_human_controls.png (control signals)
  - intersection_npace_influence_vs_human_beliefs.png  (belief convergence)
  - intersection_npace_influence_vs_human_demo.gif     (animated simulation)

Run this file from the same directory as your other intersection scripts.
"""

import os
import sys
import pathlib

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import matplotlib.patches as patches

# -----------------------------------------------------------------------------
# Paths for saving outputs next to this script
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
XY_PATH    = SCRIPT_DIR / "intersection_npace_influence_vs_human_xy.png"
CTRL_PATH  = SCRIPT_DIR / "intersection_npace_influence_vs_human_controls.png"
BEL_PATH   = SCRIPT_DIR / "intersection_npace_influence_vs_human_beliefs.png"
GIF_PATH   = SCRIPT_DIR / "intersection_npace_influence_vs_human_demo.gif"

# -----------------------------------------------------------------------------
# Extend PYTHONPATH so we can import controllers from sibling files
ROOT_DIR = SCRIPT_DIR.parent  # one level up
sys.path.append(str(ROOT_DIR))

# -----------------------------------------------------------------------------
# Imports from iLQGame and custom controllers
from iLQGame.multiplayer_dynamical_system import UncontrolledIntersection2PlayerSystem
from intersection_noisy_rational_human import IntersectNoisyRationalHuman
from intersection_npace_influence import IntersectionNPACEInfluence

# -----------------------------------------------------------------------------
# Geometry / cost parameters (unchanged from previous simulations)
R, W, L  = 70.0, 1.5, 3.0
GAMMA    = 1.0
B_PEN    = 1e5
MU       = 1e-6
V_NOM    = 16.0

DT       = 0.1
H        = 10
SIM_STEPS= 30

# Control limits
ACC_LOW, ACC_HIGH = -10.0, 10.0

# Crossing point for visualization
D_CROSS  = R / 2.0  # 35 m

# -----------------------------------------------------------------------------
# Initial conditions & true intents (unchanged)
x_curr = jnp.array([10.0, 15.0, 10.0, 15.0])  # [d1, v1, d2, v2]

THETA1_TRUE = 3  # vehicle‑1 (human) true intent
THETA2_TRUE = 1  # vehicle‑2 (robot) true intent

# -----------------------------------------------------------------------------
# Build controllers

# Human agent: noisy‑rational Q‑MDP (β=1)
human = IntersectNoisyRationalHuman(
    theta_self=THETA1_TRUE,
    ctrl_index_self=0,  # vehicle‑1 controls a1
    dt=DT, horizon=H,
    effort_w=1, b_pen=B_PEN, gamma=GAMMA,
    mu=MU, v_nom=V_NOM, R=R, W=W, L=L,
    acc_low=ACC_LOW, acc_high=ACC_HIGH,
    beta=1.0, stochastic=False, seed=42, max_iter=25
)

# Robot agent: NPACE‑Influence (teaching term)
npace_inf = IntersectionNPACEInfluence(
    theta_robot_true=THETA2_TRUE,
    dt=DT, horizon=H,
    effort_w=1.0, b_pen=B_PEN, gamma=GAMMA,
    mu=MU, v_nom=V_NOM, R=R, W=W, L=L,
    acc_low=ACC_LOW, acc_high=ACC_HIGH,
    beta_state=1, rho_forget=0.0,
    sigma2_state=(0.1, 0.1, 0.1, 0.1),
    max_iter=25, verbose=False,
    gamma_teach=1e1
  # small but nonzero teaching weight
)

# Plant dynamics
sys_dyn = UncontrolledIntersection2PlayerSystem(T=DT)

# -----------------------------------------------------------------------------
# Logs for analysis and plotting
xs_log, a1_log, a2_log = [], [], []
p_true1_log, p_true2_log = [], []  # belief in true intents

# Last observed accelerations for Bayes at t=0
a1_prev, a2_prev = 0.01, 0.01

# ---------------------------------------------------   --------------------------
# Simulation loop (time‑aligned belief updates)
for _ in range(SIM_STEPS):
    # 1) Compute controls based on last observed opponent action
    a1_cmd = human.compute_action(obs=np.asarray(x_curr), a_opponent_observed=a2_prev)
    a2_cmd = npace_inf.compute_action(obs=np.asarray(x_curr), a1_observed=a1_prev)

    # 2) Belief‑only correction: update beliefs with current opponent action
    #    We call compute_action again with the current opponent action to trigger
    #    the Bayesian update and cache update without using the returned control.
    #_ = human.compute_action(obs=np.asarray(x_curr), a_opponent_observed=a2_cmd)
    #_ = npace_inf.compute_action(obs=np.asarray(x_curr), a1_observed=a1_cmd)

    # 3) Log current state, controls, and beliefs
    xs_log.append(np.asarray(x_curr, dtype=float))
    a1_log.append(float(a1_cmd))
    a2_log.append(float(a2_cmd))

    # Human's belief about the robot's true intent (θ2_true)
    b_human = human.belief_over_theta
    p_true1_log.append(float(b_human.get(THETA2_TRUE, 0.0)))

    # Robot's belief about the human's true intent (θ1_true)
    b_robot = npace_inf.robot_belief_over_human
    p_true2_log.append(float(b_robot.get(THETA1_TRUE, 0.0)))

    # 4) Plant step: update state with commands
    x_curr = sys_dyn.disc_time_dyn(x_curr, [jnp.array([a1_cmd]), jnp.array([a2_cmd])])

    # 5) Update observed accelerations for next iteration
    a1_prev, a2_prev = a1_cmd, a2_cmd

# -----------------------------------------------------------------------------
# Convert logs to arrays for plotting
xs_arr  = np.stack(xs_log, axis=1)  # shape (4, T)
a1_arr  = np.asarray(a1_log, dtype=float)
a2_arr  = np.asarray(a2_log, dtype=float)
p1_arr  = np.asarray(p_true1_log, dtype=float)
p2_arr  = np.asarray(p_true2_log, dtype=float)
ts      = np.arange(xs_arr.shape[1], dtype=float) * DT

# -----------------------------------------------------------------------------
# XY trajectory plot (centered at crossing)
x_xy = xs_arr[2] - D_CROSS  # horizontal car (veh‑2)
y_xy = xs_arr[0] - D_CROSS  # vertical car (veh‑1)

plt.figure(figsize=(6,6))
plt.fill_between([-300, 300], -W/2,  W/2, color="#ececec", zorder=0)   # horizontal lane
plt.fill_betweenx([-300, 300], -W/2,  W/2, color="#ececec", zorder=0)  # vertical lane
plt.plot(x_xy, y_xy, "-o", ms=2.5, label="(d2 vs d1)")
plt.scatter([0], [0], c="k", s=50, marker="x", label="crossing")
plt.axis("equal"); plt.grid(True, alpha=0.3)
plt.xlabel("x = d2 − D_CROSS (m)"); plt.ylabel("y = d1 − D_CROSS (m)")
plt.title("Intersection XY: NPACE-Influence (veh‑2) vs Human (veh‑1)")
plt.legend()
plt.tight_layout(); plt.savefig(XY_PATH, dpi=150); plt.close()

# -----------------------------------------------------------------------------
# Control signals plot
plt.figure(figsize=(7,4))
plt.plot(ts, a1_arr, label="a1 (human, veh‑1)")
plt.plot(ts, a2_arr, label="a2 (NPACE-Influence, veh‑2)")
plt.xlabel("time (s)"); plt.ylabel("acceleration (m/s²)")
plt.title("Control signals – NPACE-Influence vs Human")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(CTRL_PATH, dpi=150); plt.close()

# -----------------------------------------------------------------------------
# Belief convergence plot
plt.figure(figsize=(7,4))
plt.plot(ts, p1_arr, label=f"human belief in θ2={THETA2_TRUE}")
plt.plot(ts, p2_arr, label=f"NPACE-Influence belief in θ1={THETA1_TRUE}")
plt.ylim(-0.05, 1.05)
plt.xlabel("time (s)"); plt.ylabel("belief in opponent's true intent")
plt.title("Belief convergence (NPACE-Influence vs Human)")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(BEL_PATH, dpi=150); plt.close()

# -----------------------------------------------------------------------------
# GIF drawing in meter space for readability
car_len_m, car_w_m = 3.0, 1.5
extent_m = 60.0
FIG_SIZE = (8, 8)
DPI      = 160

frames = []

def draw_frame(k: int):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_xlim(-extent_m, extent_m)
    ax.set_ylim(-extent_m, extent_m)
    ax.set_aspect('equal', adjustable='box'); ax.axis('off')

    # Roads (in meters)
    rect_h = patches.Rectangle((-extent_m, -W/2), 2*extent_m, W,
                               linewidth=0, facecolor="#d0d0d0")
    rect_v = patches.Rectangle((-W/2, -extent_m), W, 2*extent_m,
                               linewidth=0, facecolor="#d0d0d0")
    ax.add_patch(rect_h); ax.add_patch(rect_v)

    # σ-window guide lines (for visualization only)
    def left_edge(theta):   return (R / 2.0) - theta * (W / 2.0)
    def right_edge():       return ((R + W) / 2.0) + L
    yL1 = left_edge(1.0) - D_CROSS
    yL5 = left_edge(2.0) - D_CROSS
    yR  = right_edge()    - D_CROSS
    ax.axhline(yL1, color="#8888ff", ls="--", lw=1)
    ax.axhline(yL5, color="#ff8888", ls="--", lw=1)
    ax.axhline(yR,  color="#555555", ls="--", lw=1)
    ax.axvline(yL1, color="#8888ff", ls="--", lw=1)
    ax.axvline(yL5, color="#ff8888", ls="--", lw=1)
    ax.axvline(yR,  color="#555555", ls="--", lw=1)

    # Vehicle positions (centered at crossing)
    d1, v1, d2, v2 = xs_arr[:, k]
    y1 = d1 - D_CROSS  # vertical car (human)
    x2 = d2 - D_CROSS  # horizontal car (robot)

    # Vehicle 1: vertical (blue)
    car1 = patches.Rectangle((-car_w_m/2, y1 - car_len_m/2), car_w_m, car_len_m,
                             linewidth=1.0, edgecolor="black", facecolor="#1f77b4")
    ax.add_patch(car1)

    # Vehicle 2: horizontal (red)
    car2 = patches.Rectangle((x2 - car_len_m/2, -car_w_m/2), car_len_m, car_w_m,
                             linewidth=1.0, edgecolor="black", facecolor="#d62728")
    ax.add_patch(car2)

    ax.set_title(f"t = {k*DT:.2f} s | view ±{extent_m} m | crossing @ 0,0", fontsize=10)
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)
    return frame

for k in range(xs_arr.shape[1]):
    frames.append(draw_frame(k))

imageio.mimsave(GIF_PATH, frames, duration=DT)

print(f"Saved XY plot to:       {XY_PATH}")
print(f"Saved controls to:      {CTRL_PATH}")
print(f"Saved beliefs to:       {BEL_PATH}")
print(f"Saved GIF to:           {GIF_PATH}")