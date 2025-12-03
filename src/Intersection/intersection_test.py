#!/usr/bin/env python
"""
2-player Uncontrolled-Intersection ILQ test (paper-aligned)
----------------------------------------------------------
• Dynamics: ḋ = v, v̇ = a,  a ∈ [-5, 10] m/s^2
• Stage loss:  u^2  +  b · σ(d_i, θ_i) · σ(d_j, 1)
• Terminal:   -μ d_i(T) + (v_i(T) - v̄)^2,  with  v̄ = 18 m/s,  T = 3 s
• Geometry/params (from paper): R=70 m, W=3.5 m, L=4.5 m, γ=5, b=1e4

Visualization is centered at d = R/2 = 35 m (the crossing).
Saves control plot and GIF next to this file.
"""
import os, sys, pathlib
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import matplotlib.patches as patches

# ───────── paths: put outputs beside THIS file ─────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
FIG_PATH   = SCRIPT_DIR / "intersection_controls.png"
GIF_PATH   = SCRIPT_DIR / "intersection_demo.gif"

# ───────── PYTHONPATH — add src/ so we can import iLQGame ──────────
ROOT_DIR = SCRIPT_DIR.parent  # src/
sys.path.append(str(ROOT_DIR))

# ───────── Imports from solver package ──────────────────────────────
from iLQGame.cost        import UncontrolledIntersectionPlayerCost
from iLQGame.constraint  import BoxConstraint
from iLQGame.multiplayer_dynamical_system import UncontrolledIntersection2PlayerSystem
from iLQGame.ilq_solver  import ILQSolver
from iLQGame.player_cost import PlayerCost

# ───────── Paper geometry & params ─────────────────────────────────
R, W, L  = 70.0, 1.5, 3.0       # <-- corrected: R is ~70 m in the paper plots
GAMMA    = 1                # logistic sharpness
B_PEN    = 1e6                  # collision penalty scale
MU       = 10e-0                 # terminal progress weight
V_NOM    = 18.0                 # target speed
DT       = 0.1                 # 50 ms
H        = 10                   # 3 s horizon
SIM_STEPS= 30                  # 6 s rollout

# Crossing point for visualization: center the GIF at d = R/2
D_CROSS  = R / 2.0              # = 35.0 m

# σ-window edges (used both in cost and for drawing)
def left_edge(theta):   return (R / 2.0) - theta * (W / 2.0)
def right_edge():       return ((R + W) / 2.0) + L

LEFT_θ1   = left_edge(1.0)      # θ=1 left edge (≈ 33.25 m)
LEFT_θ5   = left_edge(4.0)      # θ=5 left edge (≈ 26.25 m)
RIGHT_ALL = right_edge()        # right edge (≈ 41.25 m)

# ───────── Dynamics & solver ───────────────────────────────────────
sys_dyn = UncontrolledIntersection2PlayerSystem(T=DT)

THETA1 = 1.0
THETA2 = 4.0

# Paper's test domain uses d∈[15,30] m, v∈[18,25] m/s.
# Example IC in that box (may yield small overlap initially -> small actions):
x_curr = jnp.array([20, 20.0, 20.0, 20.0])   # [d1,v1,d2,v2]

# To force visible σ overlap immediately, try:
# x_curr = jnp.array([33.5, 18.0, 34.0, 18.0])

EFFORT_W = 1.0
pc_1 = PlayerCost()
pc_2 = PlayerCost()

pc_1.add_cost(
    UncontrolledIntersectionPlayerCost(
        player_index=0, theta_self=THETA1, horizon=H,
        effort_weight=EFFORT_W, b=B_PEN, gamma=GAMMA,
        mu=MU, v_nom=V_NOM, R=R, W=W, L=L
    ),
    arg="x", weight=1.0
)
pc_2.add_cost(
    UncontrolledIntersectionPlayerCost(
        player_index=1, theta_self=THETA2, horizon=H,
        effort_weight=EFFORT_W, b=B_PEN, gamma=GAMMA,
        mu=MU, v_nom=V_NOM, R=R, W=W, L=L
    ),
    arg="x", weight=1.0
)

acc_limit_1 = BoxConstraint(-50.0, 50.0)
acc_limit_2 = BoxConstraint(-50.0, 50.0)

zeros_P = jnp.zeros((1, 4, H))
zeros_a = jnp.zeros((1,   H))

solver = ILQSolver(
    dynamics      = sys_dyn,
    player_costs  = [pc_1, pc_2],
    Ps            = [zeros_P, zeros_P],
    alphas        = [zeros_a, zeros_a],
    max_iter      = 25,
    u_constraints = [acc_limit_1, acc_limit_2],
    verbose       = False
)

# ───────── Simulation loop (MPC) ───────────────────────────────────
xs_log, a1_log, a2_log = [], [], []
for _ in range(SIM_STEPS):
    solver.run(x_curr)

    a1 = float(solver._best_operating_point[1][0][0, 0])
    a2 = float(solver._best_operating_point[1][1][0, 0]) #if False else float(solver._best_operating_point[1][1][0, 0])  # keep structure identical

    xs_log.append(x_curr)
    a1_log.append(a1)
    a2_log.append(a2)

    x_curr = sys_dyn.disc_time_dyn(x_curr, [jnp.array([a1]), jnp.array([a2])])

# ───────── Controls plot ───────────────────────────────────────────
xs_arr = np.stack([np.asarray(x) for x in xs_log], axis=1)  # (4, T)
a1_arr = np.asarray(a1_log)
a2_arr = np.asarray(a2_log)

plt.figure(figsize=(7,5))
plt.plot(a1_arr, label="a1 (veh-1)")
plt.plot(a2_arr, label="a2 (veh-2)")
plt.xlabel(f"time step (Δt = {DT:.2f}s)")
plt.ylabel("acceleration (m/s²)")
plt.title("ILQ control signals – intersection (paper-aligned R=70)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(FIG_PATH, dpi=150)

# ───────── GIF (centered at d = R/2 = 35 m) ────────────────────────
pix_per_m = 10.0
car_len_m, car_w_m = 3.0, 1.5
car_len = car_len_m * pix_per_m
car_w   = car_w_m   * pix_per_m
road_w  = W * pix_per_m
extent_m = 80.0
XMAX    = extent_m * pix_per_m

frames  = []

def draw_frame(k):
    fig, ax = plt.subplots(figsize=(6,6), dpi=100)
    ax.set_xlim(-XMAX, XMAX)
    ax.set_ylim(-XMAX, XMAX)
    ax.set_aspect('equal'); ax.axis('off')

    # Roads (center = crossing at d=R/2)
    rect_h = patches.Rectangle((-XMAX, -road_w/2), 2*XMAX, road_w, linewidth=0, facecolor="#d0d0d0")
    rect_v = patches.Rectangle((-road_w/2, -XMAX), road_w, 2*XMAX, linewidth=0, facecolor="#d0d0d0")
    ax.add_patch(rect_h); ax.add_patch(rect_v)

    # σ-window edges (for θ=1 and θ=5) mapped as (d - D_CROSS)*scale
    yL1 = (LEFT_θ1   - D_CROSS) * pix_per_m
    yL5 = (LEFT_θ5   - D_CROSS) * pix_per_m
    yR  = (RIGHT_ALL - D_CROSS) * pix_per_m
    # vertical lane (y lines)
    ax.axhline(yL1, color="#8888ff", ls="--", lw=1, label="θ=1 left" if k==0 else None)
    ax.axhline(yL5, color="#ff8888", ls="--", lw=1, label="θ=5 left" if k==0 else None)
    ax.axhline(yR,  color="#555555", ls="--", lw=1, label="right edge" if k==0 else None)
    # horizontal lane (x lines use same numeric positions)
    ax.axvline(yL1, color="#8888ff", ls="--", lw=1)
    ax.axvline(yL5, color="#ff8888", ls="--", lw=1)
    ax.axvline(yR,  color="#555555", ls="--", lw=1)

    d1, v1, d2, v2 = xs_arr[:, k]

    # Centered mapping at crossing
    y1 = (d1 - D_CROSS) * pix_per_m   # veh-1 (vertical)
    x2 = (d2 - D_CROSS) * pix_per_m   # veh-2 (horizontal)

    # Vehicle 1: vertical (blue)
    car1 = patches.Rectangle((-car_w/2, y1 - car_len/2), car_w, car_len,
                             linewidth=1.0, edgecolor="black", facecolor="#1f77b4")
    ax.add_patch(car1)

    # Vehicle 2: horizontal (red)
    car2 = patches.Rectangle((x2 - car_len/2, -car_w/2), car_len, car_w,
                             linewidth=1.0, edgecolor="black", facecolor="#d62728")
    ax.add_patch(car2)

    if k == 0:
        ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax.set_title(f"t = {k*DT:.2f} s   |   crossing @ d={D_CROSS:.2f} m")
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame

for k in range(xs_arr.shape[1]):
    frames.append(draw_frame(k))

imageio.mimsave(GIF_PATH, frames, duration=DT)

print(f"Saved control plot to: {FIG_PATH}")
print(f"Saved GIF to:         {GIF_PATH}")

# Quick preview of last frame
plt.figure(figsize=(6,5))
plt.imshow(frames[-1]); plt.axis('off'); plt.title("Final frame – intersection (R=70)")
plt.show()
