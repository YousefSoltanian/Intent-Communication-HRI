#!/usr/bin/env python
"""
2-player Planar-Navigation ILQ test (square hallway)
----------------------------------------------------
• Dynamics: unicycle heading model
      x = [x1, y1, θ1,  x2, y2, θ2]
      u0 = [v1, ω1],  u1 = [v2, ω2]
• Cost  : SocialNavigationPlayerCost (goal tracking, smooth collision barrier,
          lateral corridor comfort, speed tracking, quadratic efforts)

Outputs next to this file:
  - navigation_controls.png
  - navigation_traj.png
  - navigation_demo.gif
"""

import os, sys, pathlib
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import matplotlib.patches as patches

# ───────── paths ────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
CTRL_FIG   = SCRIPT_DIR / "navigation_controls.png"
XY_FIG     = SCRIPT_DIR / "navigation_traj.png"
GIF_PATH   = SCRIPT_DIR / "navigation_demo.gif"

# ───────── import from src/ ─────────────────────────────────────────
ROOT_DIR = SCRIPT_DIR.parent  # src/
sys.path.append(str(ROOT_DIR))

from iLQGame.multiplayer_dynamical_system import PlanarNavigation2PlayerSystem
from iLQGame.ilq_solver  import ILQSolver
from iLQGame.player_cost import PlayerCost
from iLQGame.constraint  import BoxConstraint
from iLQGame.cost        import SocialNavigationPlayerCost  # uses only theta_self

# ───────── scenario params ─────────────────────────────────────────
DT        = 0.25
H         = 10
SIM_STEPS = 40

# Make the "hallway" a square-ish arena for clearer passing behavior
HALL_LENGTH     = 10.0            # x-extent shown (±10 m)
HALL_HALF_WIDTH = 2.41            # y-extent shown (±10 m) → square

# Candidate goals (plot all; chosen goal is filled, others dashed)
goals_p0 = jnp.array([[ 4.0, +1.60],   # idx 0 (chosen)
                      [ 4.0, -1.60]])  # idx 1
goals_p1 = jnp.array([[-4.0, +1.60],   # idx 0 (chosen)
                      [-4.0, -1.60]])  # idx 1

theta0_self  = 0   # agent-1 uses goals_p0[0]
theta1_self  = 0   # agent-2 uses goals_p1[0]

# Initial joint state: two agents facing each other
x_curr = jnp.array([
    -4.0,  0.0,  0.0,     # agent-1 at left, heading →
     4.0,  0.0,  jnp.pi   # agent-2 at right, heading ←
])

# Weights (ILQ-friendly)
W_GOAL_POS   = 6.0
W_HEAD       = 1.108
W_SPEED      = 0.0
W_EFF        = 10.08
V_NOM        = 0.9

# Collision barrier (strong enough to prevent deadlock)
COL_PEN      = 15.0
COL_DIST     = 2.0

# Lateral corridor center (square arena still uses soft walls)
LAT_CENTER   = 0.0
LAT_WIDTH    = HALL_HALF_WIDTH

# Controls constraints
V_MAX, W_MAX = 1.0, 0.4
u_con_0 = BoxConstraint(jnp.array([-V_MAX,   -W_MAX]), jnp.array([V_MAX,  W_MAX]))  # allow stop, no reverse
u_con_1 = BoxConstraint(jnp.array([-V_MAX,   -W_MAX]), jnp.array([V_MAX,  W_MAX]))

# ───────── dynamics & costs ────────────────────────────────────────
sys_dyn = PlanarNavigation2PlayerSystem(T=DT)

pc_0 = PlayerCost()
pc_1 = PlayerCost()

pc_0.add_cost(
    SocialNavigationPlayerCost(
        player_index = 0,
        goals_self   = goals_p0,
        goals_other  = goals_p1,   # kept for symmetry; not used in stage cost
        theta_self   = theta0_self,
        theta_other  = 0,          # ignored by cost
        horizon      = H,
        name         = "NavCost_P0",
        w_goal_xy    = (W_GOAL_POS, W_GOAL_POS),
        w_head       = W_HEAD,
        w_speed      = W_SPEED,
        w_effort     = W_EFF,
        w_coll       = COL_PEN,
        r_safe_coll  = COL_DIST,
        hall_y0      = LAT_CENTER,
        hall_half_width = LAT_WIDTH,
        v_nom        = V_NOM
    ),
    arg="x", weight=1.0
)

pc_1.add_cost(
    SocialNavigationPlayerCost(
        player_index = 1,
        goals_self   = goals_p1,
        goals_other  = goals_p0,
        theta_self   = theta1_self,
        theta_other  = 0,          # ignored
        horizon      = H,
        name         = "NavCost_P1",
        w_goal_xy    = (W_GOAL_POS, W_GOAL_POS),
        w_head       = W_HEAD,
        w_speed      = W_SPEED,
        w_effort     = W_EFF,
        w_coll       = COL_PEN,
        r_safe_coll  = COL_DIST,
        hall_y0      = LAT_CENTER,
        hall_half_width = LAT_WIDTH,
        v_nom        = V_NOM
    ),
    arg="x", weight=1.0
)

# Warm-start seeds
zeros_P0 = jnp.zeros((2, 6, H))
zeros_P1 = jnp.zeros((2, 6, H))
zeros_a0 = jnp.zeros((2,   H))
zeros_a1 = jnp.zeros((2,   H))

solver = ILQSolver(
    dynamics      = sys_dyn,
    player_costs  = [pc_0, pc_1],
    Ps            = [zeros_P0, zeros_P1],
    alphas        = [zeros_a0, zeros_a1],
    max_iter      = 25,
    u_constraints = [u_con_0, u_con_1],
    verbose       = False,
    alpha_scaling=jnp.linspace(0.1, 2.0, 4)   
)

# ───────── simulate ────────────────────────────────────────────────
xs_log, u0_log, u1_log = [], [], []
for _ in range(SIM_STEPS):
    solver.run(x_curr)

    us = solver._best_operating_point[1]
    u0 = jnp.array([ float(us[0][0, 0]), float(us[0][1, 0]) ])   # [v1, ω1]
    u1 = jnp.array([ float(us[1][0, 0]), float(us[1][1, 0]) ])   # [v2, ω2]

    xs_log.append(x_curr)
    u0_log.append(np.asarray(u0))
    u1_log.append(np.asarray(u1))

    x_curr = sys_dyn.disc_time_dyn(x_curr, [u0, u1])

xs_arr = np.stack([np.asarray(x) for x in xs_log], axis=1)   # (6, T)
u0_arr = np.stack(u0_log, axis=0)                            # (T, 2)
u1_arr = np.stack(u1_log, axis=0)                            # (T, 2)

# ───────── controls plot ───────────────────────────────────────────
plt.figure(figsize=(8,5))
plt.subplot(2,1,1)
plt.plot(u0_arr[:,0], label="v1 (agent-1)")
plt.plot(u1_arr[:,0], label="v2 (agent-2)")
plt.ylabel("speed (m/s)"); plt.grid(True); plt.legend(loc="upper right")
plt.subplot(2,1,2)
plt.plot(u0_arr[:,1], label="ω1 (agent-1)")
plt.plot(u1_arr[:,1], label="ω2 (agent-2)")
plt.xlabel(f"time step (Δt = {DT:.2f}s)"); plt.ylabel("turn rate (rad/s)")
plt.grid(True); plt.legend(loc="upper right"); plt.tight_layout()
plt.savefig(CTRL_FIG, dpi=150)

# ───────── helpers for goal rendering ──────────────────────────────
def plot_goal_set(ax, goals_np, chosen_idx, color, label_all, label_chosen):
    """Plot all candidate goals: chosen filled, others dashed-circle."""
    # Others: dashed outline circles
    for i, (gx, gy) in enumerate(goals_np):
        if i == chosen_idx:
            continue
        circ = patches.Circle((gx, gy), radius=0.18, fill=False,
                              edgecolor=color, linestyle="--", linewidth=1.6)
        ax.add_patch(circ)
    # Chosen: filled circle
    gx, gy = goals_np[chosen_idx]
    circ = patches.Circle((gx, gy), radius=0.22, fill=True,
                          facecolor=color, edgecolor="black", linewidth=0.8,
                          label=label_chosen)
    ax.add_patch(circ)
    # Add a hollow handle just once for legend of "all candidates"
    handle = patches.Circle((goals_np[0,0], goals_np[0,1]), radius=0.18,
                            fill=False, edgecolor=color, linestyle="--", linewidth=1.6,
                            label=label_all)
    ax.add_patch(handle)

# ───────── XY trajectory plot (square arena + all goals) ──────────
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
plot_goal_set(ax, np.asarray(goals_p0), theta0_self, color="#1f77b4",
              label_all="agent-1 candidate goals", label_chosen="agent-1 chosen goal")
plot_goal_set(ax, np.asarray(goals_p1), theta1_self, color="#d62728",
              label_all="agent-2 candidate goals", label_chosen="agent-2 chosen goal")

# trajectories
plt.plot(x1, y1, "-o", ms=2, label="agent-1 traj", color="#1f77b4")
plt.plot(x2, y2, "-o", ms=2, label="agent-2 traj", color="#d62728")

plt.axis("equal"); plt.grid(True, alpha=0.3)
plt.xlim(-HALL_LENGTH/2, HALL_LENGTH/2)
plt.ylim(-HALL_HALF_WIDTH, HALL_HALF_WIDTH)
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.title("Planar navigation – trajectories with all candidate goals")
plt.legend(loc="upper right"); plt.tight_layout()
plt.savefig(XY_FIG, dpi=150)

# ───────── GIF rendering (square extents + goals each frame) ───────
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
    plot_goal_set(ax, np.asarray(goals_p0), theta0_self, color="#1f77b4",
                  label_all=None, label_chosen=None)
    plot_goal_set(ax, np.asarray(goals_p1), theta1_self, color="#d62728",
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

print(f"[ok] Saved controls plot to: {CTRL_FIG}")
print(f"[ok] Saved XY plot to     : {XY_FIG}")
print(f"[ok] Saved GIF to         : {GIF_PATH}")

# Preview last frame (optional)
plt.figure(figsize=(6.6,6.6))
plt.imshow(frames[-1]); plt.axis('off'); plt.title("Final frame – planar navigation (square)")
plt.show()
