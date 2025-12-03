#!/usr/bin/env python
"""
navigation_blame_me_vs_human_demo.py
====================================

Two-player planar navigation in a square hallway:
- Player 0: Noisy-Rational Human (vector control u=[v, ω])
- Player 1: Blame-Me/Q-MDP Robot

Matches your base sim:
  DT=0.25, H=10, SIM_STEPS=40
  Goals, weights, constraints identical
Artifacts:
  - navigation_blame_me_vs_human_xy.png
  - navigation_blame_me_vs_human_controls.png
  - navigation_blame_me_vs_human_beliefs.png
  - navigation_blame_me_vs_human_demo.gif
"""

import os, sys, pathlib
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import matplotlib.patches as patches

# -------------------------------------------------------------------
# Paths for saving outputs next to this script
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
XY_PATH    = SCRIPT_DIR / "navigation_blame_me_vs_human_xy.png"
CTRL_PATH  = SCRIPT_DIR / "navigation_blame_me_vs_human_controls.png"
BEL_PATH   = SCRIPT_DIR / "navigation_blame_me_vs_human_beliefs.png"
GIF_PATH   = SCRIPT_DIR / "navigation_blame_me_vs_human_demo.gif"

# -------------------------------------------------------------------
# Extend PYTHONPATH so we can import controllers from sibling files
ROOT_DIR = SCRIPT_DIR.parent  # one level up if your controllers live there
sys.path.append(str(ROOT_DIR))

# -------------------------------------------------------------------
# Imports from iLQGame and custom controllers
from iLQGame.multiplayer_dynamical_system import PlanarNavigation2PlayerSystem
from navigation_noisy_rational_human import NavigationNoisyRationalHuman
from navigation_blame_me_controller import NavigationBlameMeController

# -------------------------------------------------------------------
# Scenario parameters (identical to your square-hallway sim)
DT        = 0.5
H         = 10
SIM_STEPS = 40

HALL_LENGTH     = 10.0
HALL_HALF_WIDTH = 2.41  # square view (±2.41 m)

# Candidate goals (plot all; chosen goal is filled, others dashed)
goals_p0 = jnp.array([[ 4.0, +1.60],   # idx 0 (true)
                      [ 4.0, -1.60]])  # idx 1
goals_p1 = jnp.array([[-4.0, +1.60],   # idx 0 (true)
                      [-4.0, -1.60]])  # idx 1
INTENTS = (0, 1)

THETA0_TRUE = 0   # human's true goal idx
THETA1_TRUE = 0   # robot's true goal idx

# Initial joint state
x_curr = jnp.array([
    -4.0,  0.0,  0.0,     # agent-1 (human) at left, heading →
     4.0,  0.0,  jnp.pi   # agent-2 (robot) at right, heading ←
])

# Weights (ILQ-friendly, as in your sim)
W_GOAL_POS   = 60.0
W_HEAD       = 1.108
W_SPEED      = 0.0
W_EFF        = 100.08
V_NOM        = 0.9

# Collision barrier
COL_PEN      = 20.0
COL_DIST     = 2.0

# Corridor center/half-width
LAT_CENTER   = 0.0
LAT_WIDTH    = HALL_HALF_WIDTH

# Control constraints
V_MAX, W_MAX = 1.0, 0.4
V_MIN, W_MIN = -V_MAX, -W_MAX

# -------------------------------------------------------------------
# Build controllers

# Human (player 0): Noisy-Rational Q-MDP (vector controls)
human = NavigationNoisyRationalHuman(
    theta_self=THETA0_TRUE,
    ctrl_index_self=0,                      # controls player-0
    intents=INTENTS,
    # time/horizon
    dt=DT, horizon=H,
    # supply goal sets and navigation cost params
    goals_self=np.asarray(goals_p0, dtype=np.float32),
    goals_opp =np.asarray(goals_p1, dtype=np.float32),
    w_goal_xy=(W_GOAL_POS, W_GOAL_POS),
    w_head=W_HEAD, w_speed=W_SPEED, w_effort=W_EFF,
    w_lat=0.2, w_wall=60.0, w_coll=COL_PEN,
    v_nom=V_NOM, hall_y0=LAT_CENTER, hall_half_width=LAT_WIDTH,
    r_safe_coll=COL_DIST,
    # limits for [v, ω]
    v_lo=0, v_hi=V_MAX, w_lo=W_MIN, w_hi=W_MAX,
    max_iter=25, verbose=False,
    # noisy-rational sampling
    beta=0.1, stochastic=True, seed=42,
    # state-based Bayes (kept available; can be inactive depending on your impl)
    beta_state=1.0, rho_forget=0.0,
    sigma2_state=(1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1),
)

# Robot (player 1): Blame-Me/Q-MDP (vector controls)
robot = NavigationBlameMeController(
    theta_self=THETA1_TRUE,
    goals_self=np.asarray(goals_p1, dtype=np.float32),   # robot's own goals
    goals_opp =np.asarray(goals_p0, dtype=np.float32),   # opponent goal set
    ctrl_index_self=1,                                   # controls player-1
    intents=INTENTS,
    dt=DT, horizon=H,
    w_goal_xy=(W_GOAL_POS, W_GOAL_POS),
    w_head=W_HEAD, w_speed=W_SPEED, w_effort=W_EFF,
    w_lat=0.2, w_wall=60.0, w_coll=COL_PEN,
    v_nom=V_NOM, hall_y0=LAT_CENTER, hall_half_width=LAT_WIDTH,
    r_safe_coll=COL_DIST,
    v_lo=0, v_hi=V_MAX, w_lo=W_MIN, w_hi=W_MAX,
    max_iter=25, verbose=False,
    # keep extra +2λI term off (use effort in R0 from cost)
    effort_w=0.0,
    beta_action_like=0.1,
    sigma2_action_obs=(0.01, 0.01)  # e.g., 0.05 m/s & 0.05 rad/s noise
)



# Plant dynamics
sys_dyn = PlanarNavigation2PlayerSystem(T=DT)

# -------------------------------------------------------------------
# Logs
xs_log, u0_log, u1_log = [], [], []
p_human_in_theta1true, p_robot_in_theta0true = [], []

# last observed opponent controls (for Bayes at t=0)
u0_prev = np.array([0.00, 0.00], dtype=float)  # last human cmd seen by robot
u1_prev = np.array([0.00, 0.00], dtype=float)  # last robot cmd seen by human

# -------------------------------------------------------------------
# Simulation loop
for _ in range(SIM_STEPS):
    # 1) Controllers act using *last observed* opponent control
    u0_cmd = human.compute_action(
        obs=np.asarray(x_curr, dtype=float),
        a_opponent_observed=u1_prev
    )  # np.array([v, ω])
    u1_cmd = robot.compute_action(
        obs=np.asarray(x_curr, dtype=float),
        a_opponent_observed=u0_prev
    )  # np.array([v, ω])

    # 2) (Optional) belief-only correction with *current* actions
    # _ = human.compute_action(obs=np.asarray(x_curr), a_opponent_observed=u1_cmd)
    # _ = robot.compute_action(obs=np.asarray(x_curr), a_opponent_observed=u0_cmd)

    # 3) Log state, controls, beliefs
    xs_log.append(np.asarray(x_curr, dtype=float))
    u0_log.append(np.asarray(u0_cmd, dtype=float))
    u1_log.append(np.asarray(u1_cmd, dtype=float))

    # human's belief in robot's true θ
    b_h = human.belief_over_theta
    p_human_in_theta1true.append(float(b_h.get(THETA1_TRUE, 0.0)))

    # robot's belief in human's true θ
    b_r = robot.belief_over_theta
    p_robot_in_theta0true.append(float(b_r.get(THETA0_TRUE, 0.0)))

    # 4) Plant step
    x_curr = sys_dyn.disc_time_dyn(
        x_curr,
        [jnp.asarray(u0_cmd, dtype=jnp.float32),
         jnp.asarray(u1_cmd, dtype=jnp.float32)]
    )

    # 5) Expose current actions as "observed" for next step
    u0_prev, u1_prev = u0_cmd, u1_cmd

# -------------------------------------------------------------------
# Convert logs to arrays
xs_arr = np.stack(xs_log, axis=1)                    # (6, T)
u0_arr = np.stack(u0_log, axis=0)                    # (T, 2)
u1_arr = np.stack(u1_log, axis=0)                    # (T, 2)
pH_arr = np.asarray(p_human_in_theta1true, float)    # (T,)
pR_arr = np.asarray(p_robot_in_theta0true, float)    # (T,)
ts     = np.arange(xs_arr.shape[1], dtype=float) * DT

# -------------------------------------------------------------------
# Helpers (goal rendering)
def plot_goal_set(ax, goals_np, chosen_idx, color, label_all=None, label_chosen=None):
    for i, (gx, gy) in enumerate(goals_np):
        if i == chosen_idx:
            continue
        circ = patches.Circle((gx, gy), radius=0.18, fill=False,
                              edgecolor=color, linestyle="--", linewidth=1.6)
        ax.add_patch(circ)
    gx, gy = goals_np[chosen_idx]
    circ = patches.Circle((gx, gy), radius=0.22, fill=True,
                          facecolor=color, edgecolor="black", linewidth=0.8,
                          label=label_chosen)
    ax.add_patch(circ)
    if label_all:
        handle = patches.Circle((goals_np[0,0], goals_np[0,1]), radius=0.18,
                                fill=False, edgecolor=color, linestyle="--", linewidth=1.6,
                                label=label_all)
        ax.add_patch(handle)

# -------------------------------------------------------------------
# XY trajectory (square arena + all goals)
x1, y1 = xs_arr[0,:], xs_arr[1,:]
x2, y2 = xs_arr[3,:], xs_arr[4,:]

plt.figure(figsize=(7,7))
ax = plt.gca()

hall = patches.Rectangle(
    (-HALL_LENGTH/2, -HALL_HALF_WIDTH),
    HALL_LENGTH, 2*HALL_HALF_WIDTH,
    linewidth=0, facecolor="#f7f7f7"
)
ax.add_patch(hall)

plot_goal_set(ax, np.asarray(goals_p0), THETA0_TRUE, color="#1f77b4",
              label_all="human candidate goals", label_chosen="human chosen goal")
plot_goal_set(ax, np.asarray(goals_p1), THETA1_TRUE, color="#d62728",
              label_all="robot candidate goals", label_chosen="robot chosen goal")

plt.plot(x1, y1, "-o", ms=2, label="human traj", color="#1f77b4")
plt.plot(x2, y2, "-o", ms=2, label="robot traj", color="#d62728")

plt.axis("equal"); plt.grid(True, alpha=0.3)
plt.xlim(-HALL_LENGTH/2, HALL_LENGTH/2)
plt.ylim(-HALL_HALF_WIDTH, HALL_HALF_WIDTH)
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.title("Navigation XY: Blame-Me (robot) vs Noisy-Rational (human)")
plt.legend(loc="upper right"); plt.tight_layout(); plt.savefig(XY_PATH, dpi=150); plt.close()

# -------------------------------------------------------------------
# Controls plot (v and ω)
plt.figure(figsize=(8,5))
plt.subplot(2,1,1)
plt.plot(ts, u0_arr[:,0], label="v_human")
plt.plot(ts, u1_arr[:,0], label="v_robot")
plt.ylabel("speed v (m/s)"); plt.grid(True); plt.legend(loc="upper right")

plt.subplot(2,1,2)
plt.plot(ts, u0_arr[:,1], label="ω_human")
plt.plot(ts, u1_arr[:,1], label="ω_robot")
plt.xlabel(f"time (s), Δt = {DT:.2f}"); plt.ylabel("turn rate ω (rad/s)")
plt.grid(True); plt.legend(loc="upper right"); plt.tight_layout()
plt.savefig(CTRL_PATH, dpi=150); plt.close()

# -------------------------------------------------------------------
# Belief convergence
plt.figure(figsize=(7,4))
plt.plot(ts, pH_arr, label=f"human belief in robot θ={THETA1_TRUE}")
plt.plot(ts, pR_arr, label=f"robot belief in human θ={THETA0_TRUE}")
plt.ylim(-0.05, 1.05)
plt.xlabel("time (s)"); plt.ylabel("belief in opponent's true intent")
plt.title("Belief convergence: Blame-Me vs Noisy-Rational")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(BEL_PATH, dpi=150); plt.close()

# -------------------------------------------------------------------
# GIF rendering (square extents + goals each frame)
def tri_patch(x, y, th, L=0.9, W=0.45, color="#1f77b4"):
    pts = np.array([[ L/2,  0.0],
                    [-L/2,  W/2],
                    [-L/2, -W/2]])
    c, s = np.cos(th), np.sin(th)
    Rm = np.array([[c, -s],[s, c]])
    pts_w = (pts @ Rm.T) + np.array([x, y])
    return patches.Polygon(pts_w, closed=True, ec="black", fc=color, lw=1.0)

frames = []
X_EXT, Y_EXT = HALL_LENGTH/2, HALL_HALF_WIDTH
for k in range(xs_arr.shape[1]):
    fig, ax = plt.subplots(figsize=(6.6,6.6), dpi=120)
    ax.set_xlim(-X_EXT, X_EXT); ax.set_ylim(-Y_EXT, Y_EXT)
    ax.set_aspect('equal'); ax.axis('off')

    hall = patches.Rectangle((-X_EXT, -Y_EXT), 2*X_EXT, 2*Y_EXT,
                             linewidth=0, facecolor="#f7f7f7")
    ax.add_patch(hall)

    # goals each frame
    plot_goal_set(ax, np.asarray(goals_p0), THETA0_TRUE, color="#1f77b4",
                  label_all=None, label_chosen=None)
    plot_goal_set(ax, np.asarray(goals_p1), THETA1_TRUE, color="#d62728",
                  label_all=None, label_chosen=None)

    x1k, y1k, th1k = xs_arr[0,k], xs_arr[1,k], xs_arr[2,k]
    x2k, y2k, th2k = xs_arr[3,k], xs_arr[4,k], xs_arr[5,k]
    ax.add_patch(tri_patch(x1k, y1k, th1k, color="#1f77b4"))
    ax.add_patch(tri_patch(x2k, y2k, th2k, color="#d62728"))

    ax.set_title(f"t = {k*DT:.2f} s")
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    frames.append(frame)

imageio.mimsave(GIF_PATH, frames, duration=DT)

print(f"[ok] Saved XY plot to:       {XY_PATH}")
print(f"[ok] Saved controls to:      {CTRL_PATH}")
print(f"[ok] Saved beliefs to:       {BEL_PATH}")
print(f"[ok] Saved GIF to:           {GIF_PATH}")
