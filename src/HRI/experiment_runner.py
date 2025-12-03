#!/usr/bin/env python
# experiment_runner.py  – three-input Lunar-Lander runner (Aug-2025)
#
# SAMPLE COMMANDS
# ---------------
#   Warm-up (human trim keys only)……………  python experiment_runner.py warm
#   Collect 30 training trials………………  python experiment_runner.py train -n 30
#   Robot control, ILQ……………………………  python experiment_runner.py control --ctrl ilq
#   Robot control, Blame-Me…………………  python experiment_runner.py control --ctrl blame
#   Robot control, NPACE…………………………  python experiment_runner.py control --ctrl npace
#   Robot control, NPACE-Influence………  python experiment_runner.py control --ctrl influence
#
# Action tuple sent to the low-level environment:
#        ( τ_env ,  F_h (bias) ,  ΔF_r (robot) )
#
#  •  Each element may be positive *or* negative.
#  •  The environment finally clips the **sum**
#     F_tot = clip(F_h + ΔF_r , 0 , HUMAN_THRUST_MAX).
#

from __future__ import annotations
import sys, time, pickle, argparse, random
import numpy as np
import pygame, matplotlib.pyplot as plt

from goal_lander_env         import GoalLanderEnv, LANDER_HALF_HEIGHT
from ilq_thrust_controller   import ILQGameThrustController
from blame_me_controller     import BlameMeThrustController
from NPACE                    import NPACE
from NPACE_Influence         import NPACE_Influence

# ───────── simulation/UI constants ─────────────────────────────────
MAX_THRUST_HUMAN   = 2000.0        # symmetric (±)
MAX_TORQUE         = 1000.0
MAX_ANG_SP         = 3.22
MAX_SEC_PER_TRIAL  = 150.0
FPS                = 60

THRUST_SP_RATE     = 500.0         # N/s via ↑/↓
ANGLE_SP_RATE      = 0.20          # rad/s via ←/→

KP_ANGLE, KD_ANGLE = 4000.0, 1000.0
K_TRACK_THRUST, KD_TRACK_THRUST = 10.0, 0.01

HOVER_THRUST       = 800.0           # used during countdown overlay
DELTA_LIMIT        = 2000.0        # ± robot authority

# ───────── baseline PD δ-thrust controller ─────────────────────────
def pd_robot_delta_controller(obs, env, kp=20.0, kd=4.0) -> float:
    y, vy = obs[1], obs[3]
    δ     = kp*(env.goal_y_px - y) - kd*vy
    return np.clip(δ, -DELTA_LIMIT, DELTA_LIMIT)

# ───────── controller factories ────────────────────────────────────
def _fact_pd(env):        return None
def _fact_ilq(env):       return ILQGameThrustController(
                               goal_xy=(env.goal_x_px, env.goal_y_px), dt=1/FPS)
def _fact_blame(env):     return BlameMeThrustController(goal_x=env.goal_x_px)
def _fact_npace(env):     return NPACE(goal_x=env.goal_x_px)
def _fact_infl(env):      return NPACE_Influence(goal_x=env.goal_x_px)

_CONTROLLER_FACTORY = {
    "pd":        _fact_pd,
    "ilq":       _fact_ilq,
    "blame":     _fact_blame,
    "npace":     _fact_npace,
    "influence": _fact_infl,
}

# ───────── single-episode loop ─────────────────────────────────────
def run_episode(env: GoalLanderEnv, *,
                human_thrust: bool,
                ctrl_mode: str = "pd",
                log: bool = False):

    obs,_       = env.reset()
    ang_sp      = float(obs[4])
    thrust_sp   = HOVER_THRUST
    human_thr   = HOVER_THRUST if human_thrust else 0.0
    robot_thr   = 0.0
    err_prev    = 0.0

    robo_ctrl = _CONTROLLER_FACTORY[ctrl_mode](env)

    # —— prime JIT compilation during countdown only ——
    if robo_ctrl and ctrl_mode != "pd":
        try:    robo_ctrl.compute_action(obs, 0.0, 0.0)
        except TypeError:
            robo_ctrl.compute_action(obs)

    traj  = [] if log else None
    clock = pygame.time.Clock()
    t0    = time.perf_counter()

    while True:
        dt    = clock.tick(FPS)*0.001
        t_now = time.perf_counter() - t0
        if t_now > MAX_SEC_PER_TRIAL:
            break

        # ---- event handling --------------------------------------
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (
               ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit(0)

        cd_on = getattr(env, "_countdown_time", 0.0) > 0.0
        keys  = pygame.key.get_pressed()

        # ========== countdown phase: physics frozen, ignore keys ==========
        if cd_on:
            obs,_,_,_,_ = env.step((0.0, HOVER_THRUST, 0.0))
            if pygame.display.get_surface():
                env.render()
            continue
        # =================================================================

        # ---- user trims -----------------------------------------------
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            ang_sp += ANGLE_SP_RATE * dt
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            ang_sp -= ANGLE_SP_RATE * dt
        ang_sp = np.clip(ang_sp, -MAX_ANG_SP, MAX_ANG_SP)

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            thrust_sp += THRUST_SP_RATE * dt
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            thrust_sp -= THRUST_SP_RATE * dt
        thrust_sp = np.clip(thrust_sp,
                            -MAX_THRUST_HUMAN, MAX_THRUST_HUMAN)

        # ---- inner attitude loop --------------------------------------
        angle, omega = float(obs[4]), float(obs[5])
        torque_cmd   = KP_ANGLE*(ang_sp - angle) - KD_ANGLE*omega
        torque_env   = -np.clip(torque_cmd, -MAX_TORQUE, MAX_TORQUE)

        # ---- human thrust PI tracker ----------------------------------
        if human_thrust:
            err      = thrust_sp - human_thr
            d_err    = (err - err_prev) / dt
            err_prev = err
            human_thr += (K_TRACK_THRUST*err + KD_TRACK_THRUST*d_err)*dt
            human_thr  = np.clip(human_thr,
                                 -MAX_THRUST_HUMAN, MAX_THRUST_HUMAN)
        else:
            dist_y   = obs[1] - (env.goal_y_px + LANDER_HALF_HEIGHT)
            fade     = np.clip(dist_y / 150.0, 0.0, 1.0)
            human_thr = fade * HOVER_THRUST

        # ---- robot ΔF_r ----------------------------------------------
        if ctrl_mode == "pd" and human_thrust:
            robot_thr = 0.0
        elif ctrl_mode == "pd":
            robot_thr = pd_robot_delta_controller(obs, env)
        elif ctrl_mode == "ilq":
            robot_thr = robo_ctrl.compute_action(obs)
        elif ctrl_mode == "blame":
            robot_thr = robo_ctrl.compute_action(obs,
                                                 torque_env, human_thr)
        elif ctrl_mode in ("npace","influence"):
            robot_thr = robo_ctrl.compute_action(obs,
                                                 torque_env, human_thr)
        else:
            raise ValueError(f"Unknown ctrl_mode '{ctrl_mode}'")
        robot_thr = np.clip(robot_thr, -DELTA_LIMIT, DELTA_LIMIT)

        # ---- physics & render ----------------------------------------
        obs,_,done,_,_ = env.step((torque_env, human_thr, robot_thr))
        if pygame.display.get_surface():
            env.render()
        if done:
            break

        # ---- logging --------------------------------------------------
        if traj is not None:
            traj.append({"t": t_now,
                         "state":  obs.copy(),
                         "action": np.asarray((torque_env,
                                               human_thr,
                                               robot_thr), np.float32)})

    env.close()
    return (traj if traj else None), done and env._success(obs)

# ───────── diagnostics plot helper ---------------------------------
def quick_plots(sample):
    traj = sample["trajectory"]
    t    = [d["t"] for d in traj]
    τ    = [d["action"][0] for d in traj]
    Fh   = [d["action"][1] for d in traj]
    Fr   = [d["action"][2] for d in traj]
    Ft   = [h+r for h,r in zip(Fh, Fr)]
    xs   = [d["state"][0] for d in traj]
    ys   = [d["state"][1] for d in traj]
    gx,gy= sample["goal_px"]

    fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(8,7),sharex=True)
    ax1.plot(t, τ, 'r');              ax1.set_ylabel("τ (N·m)")
    ax2.plot(t, Fh,'g',label="F_h");  ax2.plot(t, Fr,'b',label="ΔF_r")
    ax2.plot(t, Ft,'k--',label="F_tot"); ax2.legend(); ax2.set_ylabel("N")
    ax3.plot(xs,ys,'-k')
    ax3.scatter(gx,gy+LANDER_HALF_HEIGHT,c="orange",marker="x",s=120)
    ax3.scatter(gx,gy,c="green",marker="_",s=200,lw=4)
    ax3.set_xlabel("x(px)"); ax3.set_ylabel("y(px)")
    ax1.set_title("Control signals"); ax3.set_title("X-Y trajectory")
    plt.tight_layout(); plt.show()

# ───────── mode helpers --------------------------------------------
def warm_up():
    print("Warm-up ▶ use ↑/↓ for thrust, ←/→ for angle.")
    env = GoalLanderEnv(show_x_goal=False, show_y_goal=True)
    run_episode(env, human_thrust=True, ctrl_mode="pd", log=False)

def training(subj:int, n_trials:int):
    fname = f"subject_{subj:02d}_training.pkl"
    print(f"Training {n_trials} successful trials → {fname}")
    data, trial = [], 1
    while len(data) < n_trials:
        print(f"  trial {trial} (target {len(data)+1}/{n_trials})")
        env = GoalLanderEnv(show_x_goal=False, show_y_goal=True)
        traj,succ = run_episode(env, human_thrust=True,
                                ctrl_mode="pd", log=True)
        if succ:
            data.append({"goal_idx":env.goal_idx,
                         "goal_px": (env.goal_x_px, env.goal_y_px),
                         "trajectory":traj})
            time.sleep(0.3)
        else:
            print("    • failed – repeating …"); time.sleep(0.5)
        trial += 1
    with open(fname,"wb") as fh: pickle.dump(data,fh)
    print("Saved →", fname); quick_plots(random.choice(data))

def control(ctrl_mode:str):
    print(f"Control ▶ ←/→ trim angle, ↑/↓ trim thrust "
          f"(robot = {ctrl_mode.upper()})")
    env = GoalLanderEnv(show_x_goal=False, show_y_goal=True)
    run_episode(env, human_thrust=True, ctrl_mode=ctrl_mode, log=False)

# ───────── CLI entry point -----------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("warm","train","control"),
                    nargs="?", default="warm")
    ap.add_argument("-s","--subj",   type=int, default=1)
    ap.add_argument("-n","--trials", type=int, default=20)
    ap.add_argument("--ctrl", choices=("pd","ilq","blame",
                                       "npace","influence"),
                    default="pd")
    args = ap.parse_args()

    if   args.mode=="warm":   warm_up()
    elif args.mode=="train":  training(args.subj, args.trials)
    else:                     control(args.ctrl)
