#!/usr/bin/env python
"""
navigation_npace_influence_vs_human_demo.py
===========================================

Two-player planar navigation simulation with:
  • Agent-1 (human): noisy-rational Q-MDP
  • Agent-2 (robot): NPACE-Influence (teaching term)

Setup mirrors your earlier navigation test (square hallway, same goals/weights/
limits) and the plotting style mirrors the intersection demo:
  - navigation_npace_influence_vs_human_xy.png        (trajectory plot)
  - navigation_npace_influence_vs_human_controls.png  (control signals)
  - navigation_npace_influence_vs_human_beliefs.png   (belief convergence)
  - navigation_npace_influence_vs_human_demo.gif      (animated simulation)
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
XY_PATH    = SCRIPT_DIR / "navigation_npace_influence_vs_human_xy.png"
CTRL_PATH  = SCRIPT_DIR / "navigation_npace_influence_vs_human_controls.png"
BEL_PATH   = SCRIPT_DIR / "navigation_npace_influence_vs_human_beliefs.png"
GIF_PATH   = SCRIPT_DIR / "navigation_npace_influence_vs_human_demo.gif"

# -----------------------------------------------------------------------------
# Extend PYTHONPATH so we can import controllers from sibling files
ROOT_DIR = SCRIPT_DIR.parent  # one level up
sys.path.append(str(ROOT_DIR))

# -----------------------------------------------------------------------------
# Imports from iLQGame and custom controllers
from iLQGame.multiplayer_dynamical_system import PlanarNavigation2PlayerSystem
from navigation_noisy_rational_human import NavigationNoisyRationalHuman
from navigation_npace_influence import NavigationNPACEInfluence

# -----------------------------------------------------------------------------
# Scenario / geometry parameters (mirrors your nav test)
DT        = 0.5
H         = 10
SIM_STEPS = 40

# Square hallway / arena (same as your previous nav script)
HALL_LENGTH     = 10.0             # x-extent shown (±10 m)
HALL_HALF_WIDTH = 2.41             # y-extent shown (±2.41 m)

# Candidate goals (plot all; chosen goal is filled, others dashed)
goals_p0 = jnp.array([[ 4.0, +1.60],   # idx 0 (chosen)
                      [ 4.0, -1.60]])  # idx 1
goals_p1 = jnp.array([[-4.0, +1.60],   # idx 0 (chosen)
                      [-4.0, -1.60]])  # idx 1

THETA1_TRUE = 0  # agent-1 (human) true goal index
THETA2_TRUE = 0  # agent-2 (robot) true goal index

# Initial joint state: two agents facing each other
x_curr = jnp.array([
    -4.0,  0.0,  0.0,     # agent-1 at left, heading →
     4.0,  0.0,  jnp.pi   # agent-2 at right, heading ←
])

# Weights (aligned with your nav sim)
W_GOAL_POS   = 60.0
W_HEAD       = 1.108
W_SPEED      = 0.0
W_EFF        = 50.08
V_NOM        = 0.9

# Collision barrier & corridor comfort
COL_PEN      = 20.0
COL_DIST     = 2.0
W_LAT        = 0.2
W_WALL       = 100.0
LAT_CENTER   = 0.0
LAT_WIDTH    = HALL_HALF_WIDTH

# Controls constraints
V_MAX, W_MAX = 1.0, 0.4
U_LOW  = jnp.array([-0*V_MAX,   -W_MAX])
U_HIGH = jnp.array([V_MAX,  W_MAX])

# -----------------------------------------------------------------------------
# Build controllers

# Human agent: noisy-rational Q-MDP (β=1), multivariate policy
human = NavigationNoisyRationalHuman(
    theta_self=THETA1_TRUE,
    ctrl_index_self=0,                      # controls player-0
    intents=(0, 1),
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
    v_lo=-0*V_MAX, v_hi=V_MAX, w_lo=-W_MAX, w_hi=W_MAX,
    max_iter=25, verbose=False,
    # noisy-rational sampling
    beta=0.1, stochastic=True, seed=42,
    # state-based Bayes (kept available; can be inactive depending on your impl)
    beta_state=1.0, rho_forget=0.0,
    sigma2_state=(1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1),
)

# Robot agent: NPACE-Influence (teaching term)
npace_inf = NavigationNPACEInfluence(
    theta_robot_true=THETA2_TRUE,
    goals_human=goals_p0, goals_robot=goals_p1,
    intents=(0, 1),
    dt=DT, horizon=H,
    w_goal_xy=(W_GOAL_POS, W_GOAL_POS),
    w_head=W_HEAD, w_speed=W_SPEED, w_effort=W_EFF,
    w_lat=W_LAT, w_wall=W_WALL, w_coll=COL_PEN,
    v_nom=V_NOM, hall_y0=LAT_CENTER, hall_half_width=LAT_WIDTH,
    r_safe_coll=COL_DIST,
    v_lo=float(U_LOW[0]), v_hi=float(U_HIGH[0]),
    w_lo=float(U_LOW[1]), w_hi=float(U_HIGH[1]),
    beta_state=1.0, rho_forget=0.0,
    # modest state noise; you can tune heading variance separately if needed
    sigma2_state=(1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1),
    max_iter=25, verbose=False,
    gamma_teach=10000.0,            # small but nonzero teaching weight
    effort_w_qmdp=0.0,           # keep zero to mirror your human Q-MDP form
    beta = 0.1                     # noisy-rationality parameter
)

# Plant dynamics
sys_dyn = PlanarNavigation2PlayerSystem(T=DT)

# -----------------------------------------------------------------------------
# Logs for analysis and plotting
xs_log, u1_log, u2_log = [], [], []
p_true1_log, p_true2_log = [], []  # belief in true intents

# Last observed controls for Bayes at t=0
u1_prev = np.array([0.0, 0.00], dtype=float)
u2_prev = np.array([0.0, 0.00], dtype=float)

# -----------------------------------------------------------------------------
# Simulation loop (time-aligned belief updates)
for _ in range(SIM_STEPS):
    # 1) Compute controls based on last observed opponent action
    u1_cmd = human.compute_action(
        obs=np.asarray(x_curr), a_opponent_observed=np.asarray(u2_prev)
    )  # array [v1, w1]
    u2_cmd = npace_inf.compute_action(
        obs=np.asarray(x_curr), a1_observed=np.asarray(u1_prev)
    )  # array [v2, w2]

    # 2) (Optional) Belief-only correction calls (kept off, same as intersection demo)
    # _ = human.compute_action(obs=np.asarray(x_curr), a_opponent_observed=np.asarray(u2_cmd))
    # _ = npace_inf.compute_action(obs=np.asarray(x_curr), a1_observed=np.asarray(u1_cmd))

    # 3) Log current state, controls, and beliefs
    xs_log.append(np.asarray(x_curr, dtype=float))
    u1_log.append(np.asarray(u1_cmd, dtype=float))
    u2_log.append(np.asarray(u2_cmd, dtype=float))

    # Human's belief about the robot's true intent (θ2_true)
    b_human = human.belief_over_theta
    p_true1_log.append(float(b_human.get(THETA2_TRUE, 0.0)))

    # Robot's belief about the human's true intent (θ1_true)
    b_robot = npace_inf.robot_belief_over_human
    p_true2_log.append(float(b_robot.get(THETA1_TRUE, 0.0)))

    # 4) Plant step: update state with commands
    x_curr = sys_dyn.disc_time_dyn(
        x_curr,
        [jnp.asarray(u1_cmd, dtype=jnp.float32),
         jnp.asarray(u2_cmd, dtype=jnp.float32)]
    )

    # 5) Update observed controls for next iteration
    u1_prev, u2_prev = u1_cmd, u2_cmd

# -----------------------------------------------------------------------------
# Convert logs to arrays for plotting
xs_arr  = np.stack(xs_log, axis=1)                     # shape (6, T)
u1_arr  = np.stack(u1_log, axis=0)                    # shape (T, 2)
u2_arr  = np.stack(u2_log, axis=0)                    # shape (T, 2)
p1_arr  = np.asarray(p_true1_log, dtype=float)        # (T,)
p2_arr  = np.asarray(p_true2_log, dtype=float)        # (T,)
ts      = np.arange(xs_arr.shape[1], dtype=float) * DT

# -----------------------------------------------------------------------------
# Controls plot
plt.figure(figsize=(8,5))
plt.subplot(2,1,1)
plt.plot(ts, u1_arr[:,0], label="v1 (human)")
plt.plot(ts, u2_arr[:,0], label="v2 (NPACE-Influence)")
plt.ylabel("speed (m/s)"); plt.grid(True, alpha=0.3); plt.legend(loc="upper right")

plt.subplot(2,1,2)
plt.plot(ts, u1_arr[:,1], label="ω1 (human)")
plt.plot(ts, u2_arr[:,1], label="ω2 (NPACE-Influence)")
plt.xlabel(f"time (s), Δt = {DT:.2f}"); plt.ylabel("turn rate (rad/s)")
plt.grid(True, alpha=0.3); plt.legend(loc="upper right"); plt.tight_layout()
plt.savefig(CTRL_PATH, dpi=150); plt.close()

# -----------------------------------------------------------------------------
# Helpers for goal rendering
def plot_goal_set(ax, goals_np, chosen_idx, color, label_all=None, label_chosen=None):
    """Plot all candidate goals: chosen filled, others dashed-circle."""
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
    # Legend handle for "all candidates"
    if label_all is not None:
        handle = patches.Circle((goals_np[0,0], goals_np[0,1]), radius=0.18,
                                fill=False, edgecolor=color, linestyle="--", linewidth=1.6,
                                label=label_all)
        ax.add_patch(handle)

# -----------------------------------------------------------------------------
# XY trajectory plot (square arena + all goals)
x1, y1 = xs_arr[0,:], xs_arr[1,:]
x2, y2 = xs_arr[3,:], xs_arr[4,:]

plt.figure(figsize=(7,7))
ax = plt.gca()

# Square arena background
hall = patches.Rectangle(
    (-HALL_LENGTH/2, -HALL_HALF_WIDTH),
    HALL_LENGTH, 2*HALL_HALF_WIDTH,
    linewidth=0, facecolor="#f7f7f7"
)
ax.add_patch(hall)

# All goals: dashed for non-chosen, filled for chosen
plot_goal_set(ax, np.asarray(goals_p0), THETA1_TRUE, color="#1f77b4",
              label_all="human candidate goals", label_chosen="human chosen goal")
plot_goal_set(ax, np.asarray(goals_p1), THETA2_TRUE, color="#d62728",
              label_all="robot candidate goals", label_chosen="robot chosen goal")

# trajectories
plt.plot(x1, y1, "-o", ms=2, label="human traj", color="#1f77b4")
plt.plot(x2, y2, "-o", ms=2, label="robot traj", color="#d62728")

plt.axis("equal"); plt.grid(True, alpha=0.3)
plt.xlim(-HALL_LENGTH/2, HALL_LENGTH/2)
plt.ylim(-HALL_HALF_WIDTH, HALL_HALF_WIDTH)
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.title("Navigation XY: NPACE-Influence (agent-2) vs Noisy-Rational Human (agent-1)")
plt.legend(loc="upper right"); plt.tight_layout()
plt.savefig(XY_PATH, dpi=150); plt.close()

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
# GIF rendering (square extents + goals each frame)
def tri_patch(x, y, th, L=0.9, W=0.45, color="#1f77b4"):
    pts = np.array([[ L/2,  0.0],
                    [-L/2,  W/2],
                    [-L/2, -W/2]])
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s],[s, c]])
    pts_w = (pts @ R.T) + np.array([x, y])
    return patches.Polygon(pts_w, closed=True, ec="black", fc=color, lw=1.0)

frames = []
X_EXT = HALL_LENGTH/2
Y_EXT = HALL_HALF_WIDTH          # same as X_EXT → square view

for k in range(xs_arr.shape[1]):
    fig, ax = plt.subplots(figsize=(6.6,6.6), dpi=120)
    ax.set_xlim(-X_EXT, X_EXT); ax.set_ylim(-Y_EXT, Y_EXT)
    ax.set_aspect('equal'); ax.axis('off')

    # arena background
    hall = patches.Rectangle((-X_EXT, -Y_EXT), 2*X_EXT, 2*Y_EXT,
                             linewidth=0, facecolor="#f7f7f7")
    ax.add_patch(hall)

    # goals (all each frame): dashed others, filled chosen
    plot_goal_set(ax, np.asarray(goals_p0), THETA1_TRUE, color="#1f77b4",
                  label_all=None, label_chosen=None)
    plot_goal_set(ax, np.asarray(goals_p1), THETA2_TRUE, color="#d62728",
                  label_all=None, label_chosen=None)

    # agents
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

print(f"Saved XY plot to:       {XY_PATH}")
print(f"Saved controls to:      {CTRL_PATH}")
print(f"Saved beliefs to:       {BEL_PATH}")
print(f"Saved GIF to:           {GIF_PATH}")
