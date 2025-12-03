#!/usr/bin/env python
"""
inverse_game_fit_v2.py – learn parametric costs for 2‑player Lunar‑Lander
using receding‑horizon ILQ simulations with surrogate gradients.
"""

import os, sys, math, pickle, random, argparse
import numpy as onp
import jax, jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from iLQGame.differentiable_extractor import (
    get_extractor,
    _make_solver,
    _SOLVER_CACHE,
)
from iLQGame.multiplayer_dynamical_system import LunarLander2PlayerSystem

lander_sys = LunarLander2PlayerSystem(T=1 / 60)

# ───────── CONFIG ────────────────────────────────────────────────────
LR            = 1e+1
ITER          = 60
INNER_ITERS   = 8
BATCH_SIZE    = 1
HORIZON_LIST  = (60, 70)
TRIM_LAST     = 5
PRINT_EVERY   = 1
ENSURE_POS    = True
VALID_SEED    = 999
w_err         = jnp.array([1.0, 1.0])          # thrust‑vs‑torque weighting
# ─────────────────────────────────────────────────────────────────────


def build_data(trajs):
    S_list, A_list, G_list = [], [], []
    for tr in trajs:
        seq  = tr["trajectory"][:-TRIM_LAST]
        S    = onp.stack([d["state"]  for d in seq])
        A    = onp.stack([d["action"] for d in seq])
        A    = onp.stack([-A[:,0], A[:,1]], axis=1)  # [torque, thrust]
        S_list.append(S); A_list.append(A); G_list.append(onp.array(tr["goal_px"]))
    return S_list, A_list, G_list


def simulate_once(θ, H, S, G):
    solver, cT, cQ = _make_solver(H)
    w, q = θ[:4], θ[4:]
    gx, gy = G
    cT.w, cT.gx, cT.gy = w, gx, gy
    cQ.q, cQ.gx, cQ.gy = q, gx, gy

    #prev_Ps = prev_alphas = None
    mods, traj = [], []; x_cur = S[0]; traj.append(onp.asarray(x_cur))

    for _ in S:
        solver.run(jnp.asarray(x_cur))

        xs, us = solver._best_operating_point
        prev_Ps, prev_alphas = solver._Ps, solver._alphas

        uT, tau = us[0][0,0], us[1][0,0]
        x_cur   = lander_sys.disc_time_dyn(x_cur,[jnp.array([uT]),jnp.array([tau])])
        mods.append(onp.asarray([uT, tau])); traj.append(onp.asarray(x_cur))

    return onp.stack(mods), onp.stack(traj)


def validate(trajs, θ, H):
    S_list, A_list, G_list = build_data(trajs)
    random.seed(VALID_SEED); idx = random.randrange(len(trajs))
    S, A, G = S_list[idx], A_list[idx], G_list[idx]
    u_mod, traj_mod = simulate_once(θ, H, S, G)
    gx, gy = G

    plt.figure(figsize=(6,4))
    plt.plot(A[:,1], label='T human'); plt.plot(u_mod[:,0],'--',label='T model')
    plt.plot(A[:,0], label='τ human'); plt.plot(u_mod[:,1],'--',label='τ model')
    plt.legend(); plt.grid(); plt.title("Controls")

    plt.figure()
    plt.plot(S[:,0],S[:,1],'k',label='human')
    plt.plot(traj_mod[:,0],traj_mod[:,1],'g',label='model')
    plt.scatter([gx],[gy],c='g',marker='x')
    plt.gca().invert_yaxis(); plt.grid(); plt.title("XY")

    plt.figure()
    plt.plot(S[:,4],'k',label='human')
    plt.plot(traj_mod[:,4],'g',label='model')
    plt.show()


def train_dataset(trajs, θ0):
    S_list, A_list, G_list = build_data(trajs)
    rng  = onp.random.default_rng(0)
    best = (math.inf, None, None)

    for H in HORIZON_LIST:
        print(f"\n=== Horizon {H} ===")
        solver, cT, cQ = _make_solver(H)
        extractor      = get_extractor(H)

        gx0, gy0 = G_list[0]; xs0 = jnp.zeros((6,H))
        us0 = [jnp.zeros((1,H)), jnp.zeros((1,H))]
        extractor(xs0, us0, jnp.asarray(θ0[:4]), jnp.asarray(θ0[4:]), gx0, gy0)

        @jax.jit
        def grad_step(th, xs, us, u_h, gx, gy):
            _, alpha0 = extractor(xs, us, th[:4], th[4:], gx, gy)
            uT, tau   = us[0][0,0], us[1][0,0]
            d         = w_err * (jnp.array([uT, tau]) - u_h)
            return jnp.vdot(alpha0, d), 0.5*jnp.sum(d**2)

        grad_value = jax.value_and_grad(grad_step, argnums=0, has_aux=True)

        θ         = jnp.asarray(θ0)
        opt       = optax.adam(LR)
        opt_state = opt.init(θ)

        for it in range(1, ITER+1):
            batch_idx = rng.choice(len(trajs), BATCH_SIZE, replace=True)

            for _ in range(INNER_ITERS):
                total_L = 0.0; total_g = onp.zeros_like(θ0)

                for idx in batch_idx:
                    S, A, (gx,gy) = S_list[idx], A_list[idx], G_list[idx]

                    #solver.reset()
                    #cT.gx = gx; cT.gy = gy; cQ.gx = gx; cQ.gy = gy
                    #cT.w,  cQ.q  = θ[:4], θ[4:]

                    prev_Ps = prev_alphas = None
                    for x, a in zip(S, A):
                        if prev_Ps is None:
                            solver.run(jnp.asarray(x))
                        else:
                            solver.run(jnp.asarray(x),
                                       Ps_warm=prev_Ps, alphas_warm=prev_alphas)
                        xs, us = solver._best_operating_point
                        prev_Ps, prev_alphas = solver._Ps, solver._alphas

                        u_h = jnp.array([a[1], a[0]])
                        (sur, mse), g = grad_value(θ, xs, us, u_h, gx, gy)
                        total_L += float(mse); total_g += onp.asarray(g)

                g = total_g / (BATCH_SIZE*len(S))
                L = total_L / (BATCH_SIZE*len(S))

                updates, opt_state = opt.update(jnp.asarray(g), opt_state, θ)
                θ = optax.apply_updates(θ, updates)
                if ENSURE_POS: θ = jnp.maximum(θ, 1e-9)

                cT.w, cQ.q = θ[:4], θ[4:]

                # ── cache‑bust & *re‑apply* weights so new solver sees θ ──
                _SOLVER_CACHE.pop(H, None)
                solver, cT, cQ = _make_solver(H)
                solver.reset()
                cT.w, cQ.q = θ[:4], θ[4:]           #  ← keep θ alive here
                #extractor  = get_extractor(H)

            if it % PRINT_EVERY == 0:
                print(f" it {it:3d}  loss {L:.6f}")
                print("    θ =", onp.asarray(θ).tolist())
                print("    g =", g.tolist())

        if L < best[0]:
            best = (L, θ, H)

    return best[1], best[2], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    θ0 = (1e-4 * onp.random.randn(8)).astype(onp.float32)
    for pkl in args.datasets:
        print(f"\n===== {pkl} =====")
        trajs = pickle.load(open(pkl,"rb"))
        θ, H, L = train_dataset(trajs, θ0)
        print(f"Best H={H}, loss={L:.6f}")
        npz = os.path.splitext(pkl)[0]+"_learned.npz"
        onp.savez(npz, theta=onp.asarray(θ), H=int(H), loss=L); print("Saved →", npz)
        if args.plot: validate(trajs, θ, H)

if __name__ == "__main__":
    main()
