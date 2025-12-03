#!/usr/bin/env python
"""
intersection_blame_me_vs_human_demo.py
======================================

Two-player intersection simulation:
  • Vehicle-1 (human): IntersectNoisyRationalHuman (noisy-rational Q-MDP)
  • Vehicle-2 (robot):  IntersectBlameMeController (Q-MDP with action-Bayes)

Setup matches your earlier NPACE-Influence vs Human demo:
  - Same dt, horizon, geometry (R,W,L), penalties, and control limits
  - Same plotting & GIF outputs, same initial state and intents

Artifacts:
  - intersection_blame_me_vs_human_xy.png
  - intersection_blame_me_vs_human_controls.png
  - intersection_blame_me_vs_human_beliefs.png
  - intersection_blame_me_vs_human_demo.gif
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
XY_PATH    = SCRIPT_DIR / "intersection_blame_me_vs_human_xy.png"
CTRL_PATH  = SCRIPT_DIR / "intersection_blame_me_vs_human_controls.png"
BEL_PATH   = SCRIPT_DIR / "intersection_blame_me_vs_human_beliefs.png"
GIF_PATH   = SCRIPT_DIR / "intersection_blame_me_vs_human_demo.gif"

# -----------------------------------------------------------------------------
# Extend PYTHONPATH so we can import controllers from sibling files
ROOT_DIR = SCRIPT_DIR.parent  # one level up
sys.path.append(str(ROOT_DIR))

# -----------------------------------------------------------------------------
# Imports from iLQGame and custom controllers
from iLQGame.multiplayer_dynamical_system import UncontrolledIntersection2PlayerSystem
from intersection_noisy_rational_human import IntersectNoisyRationalHuman
from Intersection_blame_me_final import IntersectBlameMeController

# -----------------------------------------------------------------------------
# Geometry / cost parameters (unchanged from previous simulations)
R, W, L  = 70.0, 1.5, 3.0
GAMMA    = 1
B_PEN    = 1e4
MU       = 1e-6
V_NOM    = 15.0

DT       = 0.1
H        = 10
SIM_STEPS= 30

# Control limits
ACC_LOW, ACC_HIGH = -50.0, 50.0

# Crossing point for visualization
D_CROSS  = R / 2.0  # 35 m

# -----------------------------------------------------------------------------
# Initial conditions & true intents (same as your influence demo)
x_curr = jnp.array([15.0, 15.0, 15.0, 15.0])  # [d1, v1, d2, v2]

THETA1_TRUE = 5  # vehicle-1 (human) true intent
THETA2_TRUE = 5  # vehicle-2 (robot) true intent

# -----------------------------------------------------------------------------
# Build controllers

# Human agent: noisy-rational Q-MDP (β=1), state-based Bayes
human = IntersectNoisyRationalHuman(
    theta_self=THETA1_TRUE,
    ctrl_index_self=0,  # vehicle-1 controls a1
    intents=(1, 5, 10),  # robot intents to infer
    dt=DT, horizon=H,
    effort_w=0.01, b_pen=B_PEN, gamma=GAMMA,
    mu=MU, v_nom=V_NOM, R=R, W=W, L=L,
    acc_low=ACC_LOW, acc_high=ACC_HIGH,
    max_iter=25, verbose=False,
    beta=1.0, stochastic=False, seed=42,
    beta_state=1.0, rho_forget=0.0,
    sigma2_state=(0.1, 0.1, 0.1, 0.1)
)

# Robot agent: Blame-Me (action-likelihood Bayes + Q-MDP)
robot = IntersectBlameMeController(
    theta_self=THETA2_TRUE,
    ctrl_index_self=1,         # vehicle-2 controls a2
    intents=(1, 5, 10),            # HUMAN intents to infer
    dt=DT, horizon=H,
    effort_w=0.01, b_pen=B_PEN, gamma=GAMMA,
    mu=MU, v_nom=V_NOM, R=R, W=W, L=L,
    acc_low=ACC_LOW, acc_high=ACC_HIGH,
    max_iter=25, verbose=False,
    # action-Bayes softness (σ_obs^2) + likelihood scale
    beta_action_like=1,
    sigma2_action_obs=0.001,    # keeps posteriors from becoming over-sharp
    # keep extra Q-MDP ridge off to mirror human form (can raise if needed)
    effort_w_qmdp=0.0,
    beta=1.0, stochastic=False, seed=0
)

# Plant dynamics
sys_dyn = UncontrolledIntersection2PlayerSystem(T=DT)

# -----------------------------------------------------------------------------
# Logs for analysis and plotting
xs_log, a1_log, a2_log = [], [], []
p_true1_log, p_true2_log = [], []  # belief in true intents

# Last observed accelerations for Bayes at t=0
a1_prev, a2_prev = 0.01, 0.01

# -----------------------------------------------------------------------------
# Simulation loop (time-aligned belief updates)
for _ in range(SIM_STEPS):
    # 1) Compute controls based on last observed opponent action
    a1_cmd = human.compute_action(obs=np.asarray(x_curr), a_opponent_observed=a2_prev)
    a2_cmd = robot.compute_action(obs=np.asarray(x_curr), a_opponent_observed=a1_prev)

    # 2) (Optional) Belief-only correction calls (kept off, same as other demos)
    # _ = human.compute_action(obs=np.asarray(x_curr), a_opponent_observed=a2_cmd)
    # _ = robot.compute_action(obs=np.asarray(x_curr), a_opponent_observed=a1_cmd)

    # 3) Log current state, controls, and beliefs
    xs_log.append(np.asarray(x_curr, dtype=float))
    a1_log.append(float(a1_cmd))
    a2_log.append(float(a2_cmd))

    # Human's belief about the robot's true intent (θ2_true)
    b_human = human.belief_over_theta
    p_true1_log.append(float(b_human.get(THETA2_TRUE, 0.0)))

    # Robot's belief about the human's true intent (θ1_true)
    b_robot = robot.belief_over_theta
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
x_xy = xs_arr[2] - D_CROSS  # horizontal car (veh-2)
y_xy = xs_arr[0] - D_CROSS  # vertical   car (veh-1)

plt.figure(figsize=(6,6))
plt.fill_between([-300, 300], -W/2,  W/2, color="#ececec", zorder=0)   # horizontal lane
plt.fill_betweenx([-300, 300], -W/2,  W/2, color="#ececec", zorder=0)  # vertical lane
plt.plot(x_xy, y_xy, "-o", ms=2.5, label="(d2 vs d1)")
plt.scatter([0], [0], c="k", s=50, marker="x", label="crossing")
plt.axis("equal"); plt.grid(True, alpha=0.3)
plt.xlabel("x = d2 − D_CROSS (m)"); plt.ylabel("y = d1 − D_CROSS (m)")
plt.title("Intersection XY: Blame-Me (veh-2) vs Human (veh-1)")
plt.legend()
plt.tight_layout(); plt.savefig(XY_PATH, dpi=150); plt.close()

# -----------------------------------------------------------------------------
# Control signals plot
plt.figure(figsize=(7,4))
plt.plot(ts, a1_arr, label="a1 (human, veh-1)")
plt.plot(ts, a2_arr, label="a2 (Blame-Me, veh-2)")
plt.xlabel("time (s)"); plt.ylabel("acceleration (m/s²)")
plt.title("Control signals – Blame-Me vs Human")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(CTRL_PATH, dpi=150); plt.close()

# -----------------------------------------------------------------------------
# Belief convergence plot
plt.figure(figsize=(7,4))
plt.plot(ts, p1_arr, label=f"human belief in θ2={THETA2_TRUE}")
plt.plot(ts, p2_arr, label=f"Blame-Me belief in θ1={THETA1_TRUE}")
plt.ylim(-0.05, 1.05)
plt.xlabel("time (s)"); plt.ylabel("belief in opponent's true intent")
plt.title("Belief convergence (Blame-Me vs Human)")
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
