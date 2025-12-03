#!/usr/bin/env python
"""
2-player Uncontrolled-Intersection – Human (noisy-rational) vs Blame-Me demo
============================================================================

• veh-1: IntersectNoisyRationalHuman (β=1), θ1=1
• veh-2: IntersectBlameMeController,   θ2=4
• Belief over opponent θ ∈ {1..5} with Bayesian update from observed a_opp.
• Warm-started ILQ solves internally (one per θ) each step.

Artifacts (saved next to this file):
  - intersection_human_vs_blame_xy.png
  - intersection_human_vs_blame_controls.png
  - intersection_human_vs_blame_beliefs.png
  - intersection_human_vs_blame_demo.gif
"""
import os, sys, pathlib
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import matplotlib.patches as patches

# ───────── paths: put outputs beside THIS file ─────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
XY_PATH    = SCRIPT_DIR / "intersection_human_vs_blame_xy.png"
CTRL_PATH  = SCRIPT_DIR / "intersection_human_vs_blame_controls.png"
BEL_PATH   = SCRIPT_DIR / "intersection_human_vs_blame_beliefs.png"
GIF_PATH   = SCRIPT_DIR / "intersection_human_vs_blame_demo.gif"

# ───────── PYTHONPATH — add src/ so we can import iLQGame & ctrls ──
ROOT_DIR = SCRIPT_DIR.parent  # src/
sys.path.append(str(ROOT_DIR))

# ───────── Imports ─────────────────────────────────────────────────
from iLQGame.multiplayer_dynamical_system import UncontrolledIntersection2PlayerSystem
from intersection_blame_me_controller import IntersectBlameMeController
from intersection_noisy_rational_human import IntersectNoisyRationalHuman

# ───────── Geometry / cost params (paper-aligned; unchanged) ───────
R, W, L  = 70.0, 1.5, 3.0
GAMMA    = 1
B_PEN    = 1e4
MU       = 1e-6
V_NOM    = 18.0

DT       = 0.1
H        = 30
SIM_STEPS= 30

ACC_LOW, ACC_HIGH = -50.0, 50.0

# Centering for plots
D_CROSS  = R / 2.0  # 35 m

# ───────── Initial conditions & intents (unchanged) ────────────────
x_curr = jnp.array([20.0, 20.0, 20.0, 20.0])  # [d1,v1,d2,v2]

THETA1_TRUE = 1     # veh-1 (human) true intent
THETA2_TRUE = 2     # veh-2 (robot) true intent

# ───────── Controllers ─────────────────────────────────────────────
human = IntersectNoisyRationalHuman(
    theta_self=THETA1_TRUE,
    ctrl_index_self=0,                # veh-1 controls a1
    dt=DT, horizon=H,
    effort_w=0.1, b_pen=B_PEN, gamma=GAMMA,
    mu=MU, v_nom=V_NOM, R=R, W=W, L=L,
    acc_low=ACC_LOW, acc_high=ACC_HIGH,
    beta=10.0, stochastic=True, seed=42   # β=1 noisy-rational
)
robot = IntersectBlameMeController(
    theta_self=THETA2_TRUE,
    ctrl_index_self=1,                # veh-2 controls a2
    dt=DT, horizon=H,
    effort_w=0.1, b_pen=B_PEN, gamma=GAMMA,
    mu=MU, v_nom=V_NOM, R=R, W=W, L=L,
    acc_low=ACC_LOW, acc_high=ACC_HIGH,
    max_iter=25, verbose=False
)

# ───────── Plant ───────────────────────────────────────────────────
sys_dyn = UncontrolledIntersection2PlayerSystem(T=DT)

# ───────── Logs ────────────────────────────────────────────────────
xs_log, a1_log, a2_log = [], [], []
p_true1_log, p_true2_log = [], []

# Last observed accels for Bayes at t=0
a1_prev, a2_prev = 0.0, 0.0

# ───────── Simulation loop ─────────────────────────────────────────
for _ in range(SIM_STEPS):
    # Single belief update per frame based on last observed opponent accel
    a1_cmd = human.compute_action(obs=np.asarray(x_curr), a_opponent_observed=a2_prev)
    a2_cmd = robot.compute_action(obs=np.asarray(x_curr), a_opponent_observed=a1_prev)

    # Log state, controls, beliefs
    xs_log.append(np.asarray(x_curr, dtype=float))
    a1_log.append(float(a1_cmd))
    a2_log.append(float(a2_cmd))

    b1 = human.belief_over_theta
    b2 = robot.belief_over_theta
    p_true1_log.append(float(b1.get(THETA2_TRUE, 0.0)))  # human's belief about robot's true θ
    p_true2_log.append(float(b2.get(THETA1_TRUE, 0.0)))  # robot's belief about human's true θ

    # Plant step
    x_curr = sys_dyn.disc_time_dyn(x_curr, [jnp.array([a1_cmd]), jnp.array([a2_cmd])])

    # Update "observed" accels for next Bayes step
    a1_prev, a2_prev = a1_cmd, a2_cmd

# ───────── Convert logs ────────────────────────────────────────────
xs_arr  = np.stack(xs_log, axis=1)  # (4, T)
a1_arr  = np.asarray(a1_log, dtype=float)
a2_arr  = np.asarray(a2_log, dtype=float)
p1_arr  = np.asarray(p_true1_log, dtype=float)
p2_arr  = np.asarray(p_true2_log, dtype=float)
ts      = np.arange(xs_arr.shape[1], dtype=float) * DT

# ───────── XY trajectory (x = d2 − D_CROSS, y = d1 − D_CROSS) ─────
x_xy = xs_arr[2] - D_CROSS   # horizontal car (veh-2)
y_xy = xs_arr[0] - D_CROSS   # vertical car (veh-1)

plt.figure(figsize=(7,6), dpi=160)
# Roads as thin bands around axes (in meters)
plt.fill_between([-200, 200], -W/2,  W/2, color="#ececec", zorder=0)   # horizontal lane (y≈0)
plt.fill_betweenx([-200, 200], -W/2,  W/2, color="#ececec", zorder=0)  # vertical lane   (x≈0)
plt.plot(x_xy, y_xy, "-o", ms=3.0, label="(d2 vs d1)")
plt.scatter([0],[0], c="k", s=60, marker="x", label="crossing")
plt.axis("equal"); plt.grid(True, alpha=0.35)
plt.xlim(-80, 80); plt.ylim(-80, 80)
plt.xlabel("x = d2 − D_CROSS (m)"); plt.ylabel("y = d1 − D_CROSS (m)")
plt.title("XY: Human (veh-1) vs Blame-Me (veh-2)")
plt.legend(); plt.tight_layout(); plt.savefig(XY_PATH); plt.close()

# ───────── Control plot ────────────────────────────────────────────
plt.figure(figsize=(8,4), dpi=160)
plt.plot(ts, a1_arr, label="a1 (human, veh-1)")
plt.plot(ts, a2_arr, label="a2 (blame-me, veh-2)")
plt.xlabel("time (s)"); plt.ylabel("acceleration (m/s²)")
plt.title("Control signals – human vs blame-me")
plt.grid(True, alpha=0.35); plt.legend()
plt.tight_layout(); plt.savefig(CTRL_PATH); plt.close()

# ───────── Belief plot (P[θ_true] over time) ───────────────────────
plt.figure(figsize=(8,4), dpi=160)
plt.plot(ts, p1_arr, label=f"human belief in θ2={THETA2_TRUE}")
plt.plot(ts, p2_arr, label=f"blame-me belief in θ1={THETA1_TRUE}")
plt.ylim(-0.05, 1.05)
plt.xlabel("time (s)"); plt.ylabel("belief in opponent's true intent")
plt.title("Belief convergence – human vs blame-me")
plt.grid(True, alpha=0.35); plt.legend()
plt.tight_layout(); plt.savefig(BEL_PATH); plt.close()

# ───────── GIF (meters, big/clear) ─────────────────────────────────
CAR_LEN_M, CAR_W_M = 3.0, 1.5
EXTENT_M = 80.0
frames = []

def draw_frame(k):
    fig, ax = plt.subplots(figsize=(7,7), dpi=160)
    ax.set_xlim(-EXTENT_M, EXTENT_M); ax.set_ylim(-EXTENT_M, EXTENT_M)
    ax.set_aspect('equal'); ax.axis('off')

    # Roads (meters)
    rect_h = patches.Rectangle((-EXTENT_M, -W/2), 2*EXTENT_M, W, linewidth=0, facecolor="#d0d0d0")
    rect_v = patches.Rectangle((-W/2, -EXTENT_M), W, 2*EXTENT_M, linewidth=0, facecolor="#d0d0d0")
    ax.add_patch(rect_h); ax.add_patch(rect_v)

    # σ-window guide lines (optional visuals)
    def left_edge(theta): return (R/2.0) - theta*(W/2.0)
    def right_edge():     return ((R + W) / 2.0) + L
    yL1 = left_edge(1.0) - D_CROSS; yL5 = left_edge(2.0) - D_CROSS; yR = right_edge() - D_CROSS
    for y in (yL1, yL5, yR):
        ax.axhline(y, color="#888888", ls="--", lw=0.8); ax.axvline(y, color="#888888", ls="--", lw=0.8)

    d1, v1, d2, v2 = xs_arr[:, k]
    y1 = (d1 - D_CROSS); x2 = (d2 - D_CROSS)
    # veh-1 (human): vertical blue
    car1 = patches.Rectangle((-CAR_W_M/2, y1 - CAR_LEN_M/2), CAR_W_M, CAR_LEN_M,
                             linewidth=1.2, edgecolor="black", facecolor="#1f77b4")
    # veh-2 (robot): horizontal red
    car2 = patches.Rectangle((x2 - CAR_LEN_M/2, -CAR_W_M/2), CAR_LEN_M, CAR_W_M,
                             linewidth=1.2, edgecolor="black", facecolor="#d62728")
    ax.add_patch(car1); ax.add_patch(car2)

    ax.set_title(f"t = {k*DT:.2f} s   |   crossing @ d={D_CROSS:.1f} m", fontsize=10)
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig); return frame

for k in range(xs_arr.shape[1]):
    frames.append(draw_frame(k))
imageio.mimsave(GIF_PATH, frames, duration=DT)

print(f"Saved XY plot to:  {XY_PATH}")
print(f"Saved controls to: {CTRL_PATH}")
print(f"Saved beliefs to:  {BEL_PATH}")
print(f"Saved GIF to:      {GIF_PATH}")
