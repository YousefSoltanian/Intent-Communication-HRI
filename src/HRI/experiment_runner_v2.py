#!/usr/bin/env python
# experiment_runner_v2.py  – multi-trial Lunar-Lander runner
# ----------------------------------------------------------
# * one controller instance per block (single JAX compile)
# * goal pad re-randomised every trial and pushed into controller
# * success / failure saved in logs/{subject}_{ctrl}.pkl
# ----------------------------------------------------------

from __future__ import annotations
import argparse, pathlib, pickle, random, sys, time
from typing import List, Dict, Any

import numpy as np
import pygame, matplotlib.pyplot as plt

from goal_lander_env         import GoalLanderEnv, LANDER_HALF_HEIGHT
from blame_me_controller     import BlameMeThrustController
from NPACE                    import NPACE, GX as NPACE_GX
from NPACE_Influence         import NPACE_Influence

# ---------------- simulation constants -----------------------------
FPS              = 30
HOVER_THRUST     = 00.0
MAX_THRUST_HUMAN = 2000.0
MAX_TORQUE       = 1000.0
MAX_ANG_SP       = 3.22
THRUST_SP_RATE   = 500.0
ANGLE_SP_RATE    = 0.20
KP_ANGLE, KD_ANGLE  = 4000.0, 1000.0
K_TRACK, KD_TRACK   = 10.0, 0.01
DELTA_LIMIT      = 5000.0

# ---------------- controller helpers -------------------------------
def make_controller(name: str, goal_x: float):
    if   name == "blame":     return BlameMeThrustController(goal_x=goal_x)
    elif name == "npace":     return NPACE(goal_x=goal_x)
    elif name == "influence": return NPACE_Influence(goal_x=goal_x)
    raise ValueError(name)

def update_goal(ctrl, goal_x: float):
    """Push new x-pad into an existing controller."""
    if hasattr(ctrl, "goal_x"):
        ctrl.goal_x = goal_x
    if hasattr(ctrl, "ix_true"):
        ctrl.ix_true = 0 if abs(goal_x - NPACE_GX[0]) < 1e-6 else 1
    if hasattr(ctrl, "set_goal_x"):
        ctrl.set_goal_x(goal_x)

def _belief_of(ctrl):
    if hasattr(ctrl, "belief_y"): return np.asarray([ctrl.belief_y])
    if hasattr(ctrl, "belief"):     return np.asarray(ctrl.belief)
    return None

# ---------------- single-trial loop -------------------------------
def run_trial(env: GoalLanderEnv, ctrl, ctrl_name: str, *, human_thrust=True):
    obs, _ = env.reset()                       # new random pad
    update_goal(ctrl, env.goal_x_px)

    # JIT warm-up once
    if not getattr(ctrl, "_compiled", False):
        try:    ctrl.compute_action(obs, 0.0, 0.0)
        except TypeError:
            try: ctrl.compute_action(obs)
            except Exception: pass
        ctrl._compiled = True                  # type: ignore[attr-defined]

    ang_sp, thr_sp = float(obs[4]), HOVER_THRUST
    F_h, err_prev  = (HOVER_THRUST if human_thrust else 0.0), 0.0

    step_log: List[Dict[str, Any]] = []
    clock   = pygame.time.Clock()
    t0      = time.perf_counter()

    while True:
        dt    = clock.tick_busy_loop(FPS) * 0.001
        t_now = time.perf_counter() - t0

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (
               ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit(0)

        # countdown overlay
        if env._countdown_time > 0:
            obs, _, _, _, _ = env.step((0.0, HOVER_THRUST, 0.0))
            if pygame.display.get_surface():
                env.render()
            continue

        # user trims
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            ang_sp += ANGLE_SP_RATE * dt
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            ang_sp -= ANGLE_SP_RATE * dt
        ang_sp = np.clip(ang_sp, -MAX_ANG_SP, MAX_ANG_SP)

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            thr_sp += THRUST_SP_RATE * dt
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            thr_sp -= THRUST_SP_RATE * dt
        thr_sp = np.clip(thr_sp, -MAX_THRUST_HUMAN, MAX_THRUST_HUMAN)

        # attitude loop
        angle, omega = float(obs[4]), float(obs[5])
        τ_cmd = KP_ANGLE * (ang_sp - angle) - KD_ANGLE * omega
        τ_env = -np.clip(τ_cmd, -MAX_TORQUE, MAX_TORQUE)

        # human thrust PI
        if human_thrust:
            err   = thr_sp - F_h
            d_err = (err - err_prev) / dt
            err_prev = err
            F_h += (K_TRACK * err + KD_TRACK * d_err) * dt
            F_h  = np.clip(F_h, -MAX_THRUST_HUMAN, MAX_THRUST_HUMAN)
        else:
            dist_y = obs[1] - (env.goal_y_px + LANDER_HALF_HEIGHT)
            fade   = np.clip(dist_y / 150.0, 0.0, 1.0)
            F_h    = fade * HOVER_THRUST

        # robot ΔF_r
        if ctrl_name in ("blame", "npace", "influence"):
            try:    δF_r = ctrl.compute_action(obs, τ_env, F_h)
            except TypeError:
                δF_r = ctrl.compute_action(obs)
        else:
            δF_r = 0.0
        δF_r = np.clip(δF_r, -DELTA_LIMIT, DELTA_LIMIT)

        # physics & render
        obs, _, done, _, _ = env.step((τ_env, F_h, δF_r))
        if pygame.display.get_surface():
            env.render()

        # log step
        step_log.append({
            "t": t_now,
            "state": obs.copy(),
            "tau": τ_env,
            "F_h": F_h,
            "delta_F_r": δF_r,
            "belief": _belief_of(ctrl)
        })

        if done:
            break

    return {
        "goal_idx": env.goal_idx,
        "goal_px":  (env.goal_x_px, env.goal_y_px),
        "outcome":  env.outcome,
        "trajectory": step_log,
    }

# ---------------- quick plot ---------------------------------------
def quick_plot(sample):
    xs = [s["state"][0] for s in sample["trajectory"]]
    ys = [s["state"][1] for s in sample["trajectory"]]
    gx, gy = sample["goal_px"]
    plt.figure(figsize=(5, 5))
    plt.plot(xs, ys, "-k")
    plt.scatter(gx, gy + LANDER_HALF_HEIGHT, c="orange", marker="x", s=100)
    plt.scatter(gx, gy, c="green", marker="_", s=180, lw=3)
    plt.gca().invert_yaxis()
    plt.title(sample["outcome"])
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout(); plt.show()

# ---------------- run N trials -------------------------------------
def run_block(subject: str, ctrl_name: str, n_trials: int):
    pathlib.Path("logs").mkdir(exist_ok=True)
    fname = f"logs/{subject}_{ctrl_name}.pkl"

    env  = GoalLanderEnv(show_x_goal=False, show_y_goal=True)
    env.reset()
    ctrl = make_controller(ctrl_name, env.goal_x_px)

    trials: List[Dict[str, Any]] = []
    for k in range(1, n_trials + 1):
        print(f"{ctrl_name.upper()}  trial {k}/{n_trials} … ", end="", flush=True)
        sample = run_trial(env, ctrl, ctrl_name, human_thrust=True)
        trials.append(sample)
        print(f"{sample['outcome']}  (goal={sample['goal_idx']})")
        time.sleep(0.05)

    with open(fname, "wb") as fh:
        pickle.dump(trials, fh)
    print("Saved →", fname)
    quick_plot(random.choice(trials))
    env.close()

# ---------------- CLI ----------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", required=True, help="subject ID (e.g. s1)")
    ap.add_argument("--ctrl", nargs="+", required=True,
                    choices=("blame", "npace", "influence"))
    ap.add_argument("--trials", type=int, default=20,
                    help="trials per controller (default 20)")
    args = ap.parse_args()

    for ctrl in args.ctrl:
        run_block(args.subj, ctrl, args.trials)
