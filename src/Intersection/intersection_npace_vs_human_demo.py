#!/usr/bin/env python
"""
2-player Uncontrolled-Intersection – NPACE (robot) vs Human (noisy-rational) demo
=================================================================================

• veh-1 (human): IntersectNoisyRationalHuman (β=1, stochastic)
• veh-2 (robot): IntersectionNPACE (NPACE-style nested Q-MDP)

SAME setup as your previous scripts:
  DT=0.1, H=30, SIM_STEPS=120, γ=0.5, b=1e4, μ=1e-6, v_nom=18,
  R=70, W=1.5, L=3.0, accel limits [-500, 1000], IC [16,20,16,20],
  belief time-alignment (second pass), XY = (d2-D_CROSS, d1-D_CROSS),
  GIF pixel pipeline (pix_per_m=10, extent_m=250, 6x6in, dpi=100).

Artifacts (saved next to this file):
  - intersection_npace_vs_human_xy.png
  - intersection_npace_vs_human_controls.png
  - intersection_npace_vs_human_beliefs.png
  - intersection_npace_vs_human_demo.gif
"""
import os, sys, pathlib

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import matplotlib.patches as patches
import jax
from jax import random

OBS_NOISE_STD = 10.0   # m/s^2, std dev of accel observation noise
OBS_NOISE_CLIP = 5.0  # optional: cap extreme noise
rng = random.PRNGKey(0)  # reproducible seed

# ───────── paths: put outputs beside THIS file ─────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
XY_PATH    = SCRIPT_DIR / "intersection_npace_vs_human_xy.png"
CTRL_PATH  = SCRIPT_DIR / "intersection_npace_vs_human_controls.png"
BEL_PATH   = SCRIPT_DIR / "intersection_npace_vs_human_beliefs.png"
GIF_PATH   = SCRIPT_DIR / "intersection_npace_vs_human_demo.gif"

# ───────── PYTHONPATH — add src/ so we can import iLQGame & ctrls ──
ROOT_DIR = SCRIPT_DIR.parent  # src/
sys.path.append(str(ROOT_DIR))

# ───────── Imports ─────────────────────────────────────────────────
from iLQGame.multiplayer_dynamical_system import UncontrolledIntersection2PlayerSystem
from intersection_noisy_rational_human import IntersectNoisyRationalHuman
from intersection_npace import IntersectionNPACE

# ───────── Geometry / cost params (paper-aligned; UNCHANGED) ───────
R, W, L  = 70.0, 1.5, 3.0
GAMMA    = 1
B_PEN    = 1e3
MU       = 1e-6
V_NOM    = 18.0

DT       = 0.1
H        = 30
SIM_STEPS= 30

# Baseline wide limits (UNCHANGED)
ACC_LOW, ACC_HIGH = -50.0, 50.0

# Centering for plots
D_CROSS  = R / 2.0  # 35 m

# ───────── Initial conditions & intents (UNCHANGED) ────────────────
x_curr = jnp.array([20.0, 20.0, 20.0, 20.0])  # [d1,v1,d2,v2]

THETA1_TRUE = 1     # veh-1 (human) true intent
THETA2_TRUE = 4    # veh-2 (robot) true intent

# ───────── Controllers (parameters UNCHANGED) ──────────────────────
human = IntersectNoisyRationalHuman(
    theta_self=THETA1_TRUE,
    ctrl_index_self=0,                # veh-1 controls a1
    dt=DT, horizon=H,
    effort_w=1.0, b_pen=B_PEN, gamma=GAMMA,
    mu=MU, v_nom=V_NOM, R=R, W=W, L=L,
    acc_low=ACC_LOW, acc_high=ACC_HIGH,
    beta=10.0, stochastic=True, seed=42
)

npace = IntersectionNPACE(
    theta_robot_true=THETA2_TRUE,     # robot's own θ_r
    dt=DT, horizon=H,
    effort_w=1.0, b_pen=B_PEN, gamma=GAMMA,
    mu=MU, v_nom=V_NOM, R=R, W=W, L=L,
    acc_low=ACC_LOW, acc_high=ACC_HIGH,
    beta_state=1.0, rho_forget=0.0,  # modest state-Bayes settings
    sigma2_state=(25000.0**1, 40000.0**1, 36.0**1, 1.0**1),
    max_iter=25, verbose=False
)

# ───────── Plant ───────────────────────────────────────────────────
sys_dyn = UncontrolledIntersection2PlayerSystem(T=DT)

# ───────── Logs ────────────────────────────────────────────────────
xs_log, a1_log, a2_log = [], [], []
p_true1_log, p_true2_log = [], []   # human's P[θ2_true], robot's P[θ1_true]

# Last observed accels (t=0 Bayes)
a1_prev, a2_prev = 0.01, 0.01

# ───────── Simulation loop (belief time-aligned, UNCHANGED) ────────
for _ in range(SIM_STEPS):
    # 1) Compute controls using LAST observed opponent accel
    a1_cmd = human.compute_action(obs=np.asarray(x_curr), a_opponent_observed=a2_prev)
    a2_cmd = npace.compute_action(obs=np.asarray(x_curr), a1_observed=a1_prev)

    # 2) Log state, controls, beliefs
    xs_log.append(np.asarray(x_curr, dtype=float))
    a1_log.append(float(a1_cmd))
    a2_log.append(float(a2_cmd))

    # human's belief over robot θ_r:
    b_human = human.belief_over_theta                 # dict {θ:prob}
    p_true1_log.append(float(b_human.get(THETA2_TRUE, 0.0)))

    # robot's belief over human θ_h:
    b_robot = npace.robot_belief_over_human           # dict {θ:prob}
    p_true2_log.append(float(b_robot.get(THETA1_TRUE, 0.0)))

    # 3) Plant step
    x_curr = sys_dyn.disc_time_dyn(x_curr, [jnp.array([a1_cmd]), jnp.array([a2_cmd])])


    rng, k1, k2 = random.split(rng, 3)
    n1 = jnp.clip(OBS_NOISE_STD * random.normal(k1, ()), -OBS_NOISE_CLIP, OBS_NOISE_CLIP)
    n2 = jnp.clip(OBS_NOISE_STD * random.normal(k2, ()), -OBS_NOISE_CLIP, OBS_NOISE_CLIP)
    # 4) Update "observed" accels for next Bayes step
    a1_prev, a2_prev = a1_cmd+0*n1, a2_cmd+0*n2

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

plt.figure(figsize=(6,6))
# Roads as thin bands around axes
plt.fill_between([-300, 300], -W/2,  W/2, color="#ececec", zorder=0)   # horizontal lane (y≈0)
plt.fill_betweenx([-300, 300], -W/2,  W/2, color="#ececec", zorder=0)  # vertical lane   (x≈0)
plt.plot(x_xy, y_xy, "-o", ms=2.5, label="(d2 vs d1)")
plt.scatter([0],[0], c="k", s=50, marker="x", label="crossing")
plt.axis("equal"); plt.grid(True, alpha=0.3)
plt.xlabel("x = d2 − D_CROSS (m)"); plt.ylabel("y = d1 − D_CROSS (m)")
plt.title("Intersection XY: NPACE (veh-2) vs Human (veh-1)")
plt.legend()
plt.tight_layout(); plt.savefig(XY_PATH, dpi=150); plt.close()

# ───────── Control plot ────────────────────────────────────────────
plt.figure(figsize=(7,4))
plt.plot(ts, a1_arr, label="a1 (human, veh-1)")
plt.plot(ts, a2_arr, label="a2 (NPACE, veh-2)")
plt.xlabel("time (s)"); plt.ylabel("acceleration (m/s²)")
plt.title("Control signals")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(CTRL_PATH, dpi=150); plt.close()

# ───────── Belief plot (ADD THIS) ──────────────────────────────────
plt.figure(figsize=(7,4))
plt.plot(ts, p1_arr, label=f"human belief P[θ_r={THETA2_TRUE}]")
plt.plot(ts, p2_arr, label=f"robot belief P[θ_h={THETA1_TRUE}]")
plt.ylim(-0.05, 1.05)
plt.xlabel("time (s)")
plt.ylabel("belief")
plt.title("Belief convergence")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(BEL_PATH, dpi=150); plt.close()
print(f"Saved belief plot to:   {BEL_PATH}")

# ───────── GIF (meter-based drawing so cars aren’t tiny) ───────────
# Draw directly in meters: axis limits control visual scale.
car_len_m, car_w_m = 3.0, 1.5        # car footprint (m) — unchanged
extent_m = 60.0                      # half-width of view (m) → zoomed
FIG_SIZE = (8, 8)                    # bigger figure
DPI      = 160                       # crisper frames

def left_edge(theta):   return (R / 2.0) - theta * (W / 2.0)
def right_edge():       return ((R + W) / 2.0) + L

LEFT_θ1   = left_edge(1.0)
LEFT_θ5   = left_edge(2.0)           # proxy for widest band shown
RIGHT_ALL = right_edge()

frames = []

def draw_frame(k):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_xlim(-extent_m, extent_m)
    ax.set_ylim(-extent_m, extent_m)
    ax.set_aspect('equal', adjustable='box'); ax.axis('off')

    # Roads (in meters, centered on crossing at (0,0))
    rect_h = patches.Rectangle((-extent_m, -W/2), 2*extent_m, W,
                               linewidth=0, facecolor="#d0d0d0")
    rect_v = patches.Rectangle((-W/2, -extent_m), W, 2*extent_m,
                               linewidth=0, facecolor="#d0d0d0")
    ax.add_patch(rect_h); ax.add_patch(rect_v)

    # σ-window guide lines (in meters)
    yL1 = LEFT_θ1   - D_CROSS
    yL5 = LEFT_θ5   - D_CROSS
    yR  = RIGHT_ALL - D_CROSS
    ax.axhline(yL1, color="#8888ff", ls="--", lw=1)
    ax.axhline(yL5, color="#ff8888", ls="--", lw=1)
    ax.axhline(yR,  color="#555555", ls="--", lw=1)
    ax.axvline(yL1, color="#8888ff", ls="--", lw=1)
    ax.axvline(yL5, color="#ff8888", ls="--", lw=1)
    ax.axvline(yR,  color="#555555", ls="--", lw=1)

    # Vehicle poses (in meters)
    d1, v1, d2, v2 = xs_arr[:, k]
    y1 = d1 - D_CROSS         # veh-1 (vertical road)
    x2 = d2 - D_CROSS         # veh-2 (horizontal road)

    # Vehicle 1: vertical (blue)
    car1 = patches.Rectangle((-car_w_m/2, y1 - car_len_m/2),
                             car_w_m, car_len_m,
                             linewidth=1.0, edgecolor="black", facecolor="#1f77b4")
    ax.add_patch(car1)

    # Vehicle 2: horizontal (red)
    car2 = patches.Rectangle((x2 - car_len_m/2, -car_w_m/2),
                             car_len_m, car_w_m,
                             linewidth=1.0, edgecolor="black", facecolor="#d62728")
    ax.add_patch(car2)

    ax.set_title(f"t = {k*DT:.2f} s   |   view ±{extent_m} m   |   crossing @ 0,0")
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)
    return frame

for k in range(xs_arr.shape[1]):
    frames.append(draw_frame(k))

imageio.mimsave(GIF_PATH, frames, duration=DT)

print(f"Saved GIF to:           {GIF_PATH}")

# Quick preview of last frame if running interactively
if __name__ == "__main__":
    plt.figure(figsize=(6,5))
    plt.imshow(frames[-1]); plt.axis('off'); plt.title("Final frame – NPACE vs Human (zoomed)")
    plt.show()

