#!/usr/bin/env python
"""
Quick 3-player Lunar-Lander ILQ test
====================================

Players
-------
0 : F_h  (human thrust)      – may be ±1000 N
1 : F_r  (robot Δ-thrust)    – may be ±1000 N
2 : τ    (torque, CCW +)     – may be ±1000 N·m

The dynamics clip the *total* thrust F_h+F_r so it never goes negative.

Run from src/HRI:
    python Lunar_ilq_test.py
"""
import os, sys
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# ───────── PYTHONPATH — add src/ ────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)                         # parent of iLQGame/

# ───────── Imports from solver package ──────────────────────────────
from iLQGame.cost        import ThrustPlayerParamCost, TorquePlayerParamCost
from iLQGame.constraint  import BoxConstraint
from iLQGame.multiplayer_dynamical_system import LunarLander3PlayerSystem
from iLQGame.ilq_solver  import ILQSolver
from iLQGame.player_cost import PlayerCost

# ───────── Dynamics ────────────────────────────────────────────────
DT         = 1/60
lander_sys = LunarLander3PlayerSystem(T=DT)

# ───────── Simulation parameters ───────────────────────────────────
SIM_STEPS  = 500          # ~8.3 s
H          = 60           # 1-s horizon

x_curr = jnp.array([0., 0., 0., 0., 0., 0.])   # initial COM high
GX, GY = 400., 500.                            # target COM

# ───────── Weights (state-tracking 4 + effort 1) ───────────────────
state_w = jnp.array([968, 962, 1036, 1035], dtype=jnp.float32)
T_eff_h = 1.0      # effort weight for human thrust
T_eff_r = 1.0      # effort weight for robot thrust
tau_eff = 9.0      # effort weight for torque (same ratio as before)

w_h   = jnp.concatenate([state_w, jnp.array([T_eff_h])])
w_r   = jnp.concatenate([state_w, jnp.array([T_eff_r])])
q_tau = jnp.concatenate([state_w, jnp.array([tau_eff])])

# ───────── Cost objects & PlayerCost wrappers ──────────────────────
pc_h   = PlayerCost()
pc_r   = PlayerCost()
pc_tau = PlayerCost()

pc_h.add_cost(
    ThrustPlayerParamCost(w_h, (GX, GY), horizon=H),
    arg="x", weight=1.0
)
pc_r.add_cost(
    ThrustPlayerParamCost(w_r, (GX, GY), horizon=H),
    arg="x", weight=1.0
)
pc_tau.add_cost(
    TorquePlayerParamCost(q_tau, (GX, GY), horizon=H),
    arg="x", weight=1.0
)

# ───────── Input constraints ───────────────────────────────────────
thr_limit = BoxConstraint(-1000.0, 1000.0)   # both thrust channels
tor_limit = BoxConstraint(-1000.0, 1000.0)

# ───────── Seed gains / feeds (zeros) ──────────────────────────────
zeros_P = jnp.zeros((1, 6, H))
zeros_a = jnp.zeros((1,   H))

solver = ILQSolver(
    dynamics      = lander_sys,
    player_costs  = [pc_h, pc_r, pc_tau],
    Ps            = [zeros_P, zeros_P, zeros_P],
    alphas        = [zeros_a, zeros_a, zeros_a],
    max_iter      = 15,
    u_constraints = [thr_limit, thr_limit, tor_limit],
    verbose       = False
)

# ───────── Simulation loop ─────────────────────────────────────────
xs_log, Fh_log, Fr_log, Ft_log, tau_log = [], [], [], [], []

for _ in range(SIM_STEPS):
    solver.run(x_curr)

    F_h  = float(solver._best_operating_point[1][0][0, 0])  # player-0
    F_r  = float(solver._best_operating_point[1][1][0, 0])  # player-1
    τ    = float(solver._best_operating_point[1][2][0, 0])  # player-2
    F_tot= F_h + F_r                                        # may be clipped in dyn

    xs_log.append(x_curr)
    Fh_log.append(F_h)
    Fr_log.append(F_r)
    Ft_log.append(F_tot)
    tau_log.append(τ)

    x_curr = lander_sys.disc_time_dyn(
        x_curr,
        [jnp.array([F_h]), jnp.array([F_r]), jnp.array([τ])]
    )

# ───────── Plot results ────────────────────────────────────────────
xs_arr = np.stack([np.asarray(x) for x in xs_log], axis=1)
Fh_arr = np.asarray(Fh_log)
Fr_arr = np.asarray(Fr_log)
Ft_arr = np.asarray(Ft_log)
tau_arr= np.asarray(tau_log)

plt.figure()
plt.plot(xs_arr[0], xs_arr[1], "-o", label="COM path")
plt.scatter([GX], [GY], c="red", marker="x", s=100, label="Goal")
plt.gca().invert_yaxis()
plt.xlabel("x (px)"); plt.ylabel("y (px)")
plt.title("Lunar-Lander trajectory"); plt.legend(); plt.grid(True)

plt.figure(figsize=(7,5))
plt.plot(Fh_arr, label="F_h (human)")
plt.plot(Fr_arr, label="F_r (robot Δ)")
plt.plot(Ft_arr, '--', label="F_total")
plt.plot(tau_arr, label="τ (torque)")
plt.xlabel(f"time step (Δt = {DT:.3f}s)")
plt.ylabel("control")
plt.title("ILQ control signals"); plt.legend(); plt.grid(True)

plt.tight_layout(); plt.show()
