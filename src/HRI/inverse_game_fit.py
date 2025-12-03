#!/usr/bin/env python
"""
inverse_game_fit_v3.py – learn parametric costs for 2‑player Lunar‑Lander
with surrogate gradients (state weights + torque effort weight).

θ layout  (length 9)
────────────────────
  θ[0:4]  → thrust state weights      w₁…w₄
  θ[4:8]  → torque state weights      q₁…q₄
  θ[8]    → torque effort weight λτ   (learned)
  thrust effort weight λT ≡ 1.0 (fixed)
"""

import os, sys, math, pickle, random, argparse
import numpy as onp
import jax, jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from iLQGame.differentiable_extractor import (
    get_extractor, _make_solver, _SOLVER_CACHE
)
from iLQGame.multiplayer_dynamical_system import LunarLander2PlayerSystem

lander_sys = LunarLander2PlayerSystem(T=1 / 60)

# ───────── CONFIG ──────────────────────────────────────────────────
LR            = 1e-0
ITER          = 40
INNER_ITERS   = 5
BATCH_SIZE    = 1
HORIZON_LIST  = (30, 60, 90)
TRIM_LAST     = 5
PRINT_EVERY   = 1
ENSURE_POS    = True
VALID_SEED    = 999
w_err         = jnp.array([1.0, 1.0])          # thrust vs torque scaling
# ────────────────────────────────────────────────────────────────────


# -------------------------------------------------------------------
def _split_theta(th: jnp.ndarray):
    """Return (w_vec[5], q_vec[5]) where last entry is the effort weight."""
    w_vec = jnp.concatenate([th[:4], jnp.array([1.0], th.dtype)])   # λT fixed
    q_vec = jnp.concatenate([th[4:8], th[8:9]])                     # λτ learned
    return w_vec, q_vec


# -------------------------------------------------------------------
def build_data(trajs):
    S_ls, A_ls, G_ls = [], [], []
    for tr in trajs:
        seq  = tr["trajectory"][:-TRIM_LAST]
        S    = onp.stack([d["state"]  for d in seq])
        Araw = onp.stack([d["action"] for d in seq])
        A    = onp.stack([-Araw[:,0], Araw[:,1]], axis=1)  # [τ, T]
        S_ls.append(S); A_ls.append(A); G_ls.append(onp.array(tr["goal_px"]))
    return S_ls, A_ls, G_ls


# -------------------------------------------------------------------
def simulate_once(θ, H, S, G):
    solver, cT, cQ = _make_solver(H)
    w_vec, q_vec    = _split_theta(θ)
    gx, gy          = G
    cT.w, cT.gx, cT.gy = w_vec, gx, gy
    cQ.q, cQ.gx, cQ.gy = q_vec, gx, gy

    mods, traj = [], []
    x_cur      = S[0]; traj.append(onp.asarray(x_cur))

    for _ in S:
        solver.run(jnp.asarray(x_cur))
        xs, us  = solver._best_operating_point
        uT, τ   = us[0][0,0], us[1][0,0]
        x_cur   = lander_sys.disc_time_dyn(x_cur,
                                           [jnp.array([uT]), jnp.array([τ])])
        mods.append(onp.asarray([uT, τ]))
        traj.append(onp.asarray(x_cur))

    return onp.stack(mods), onp.stack(traj)


# -------------------------------------------------------------------
def validate(trajs, θ, H):
    S_ls, A_ls, G_ls = build_data(trajs)
    random.seed(VALID_SEED)
    idx               = random.randrange(len(trajs))
    S, A, G           = S_ls[idx], A_ls[idx], G_ls[idx]
    u_mod, traj_mod   = simulate_once(θ, H, S, G)
    gx, gy            = G

    plt.figure(figsize=(6,4))
    plt.plot(A[:,1],           label='T human')
    plt.plot(u_mod[:,0],'--',  label='T model')
    plt.plot(A[:,0],           label='τ human')
    plt.plot(u_mod[:,1],'--',  label='τ model')
    plt.legend(); plt.grid(); plt.title("Controls")

    plt.figure()
    plt.plot(S[:,0], S[:,1], 'k', label='human')
    plt.plot(traj_mod[:,0], traj_mod[:,1], 'g', label='model')
    plt.scatter([gx],[gy], c='g', marker='x')
    plt.gca().invert_yaxis(); plt.grid(); plt.title("XY"); plt.legend()

    plt.figure()
    plt.plot(S[:,4],      'k', label='θ human')
    plt.plot(traj_mod[:,4],'g', label='θ model')
    plt.legend(); plt.show()


# -------------------------------------------------------------------
def train_dataset(trajs, θ0):
    S_ls, A_ls, G_ls = build_data(trajs)
    rng   = onp.random.default_rng(0)
    best  = (math.inf, None, None)

    for H in HORIZON_LIST:
        print(f"\n=== Horizon {H} ===")
        solver, cT, cQ = _make_solver(H)
        extractor      = get_extractor(H)

        # warm‑up compile
        gx0, gy0   = G_ls[0]
        xs0        = jnp.zeros((6,H))
        us0        = [jnp.zeros((1,H)), jnp.zeros((1,H))]
        w0, q0     = _split_theta(jnp.asarray(θ0))
        extractor(xs0, us0, w0, q0, gx0, gy0)

        @jax.jit
        def grad_step(th, xs, us, u_h, gx, gy):
            w_vec, q_vec        = _split_theta(th)
            _, alpha0           = extractor(xs, us, w_vec, q_vec, gx, gy)
            uT, τ               = us[0][0,0], us[1][0,0]
            d                   = w_err * (jnp.array([uT, τ]) - u_h)
            return jnp.vdot(alpha0, d), 0.5*jnp.sum(d**2)

        value_grad = jax.value_and_grad(grad_step, argnums=0, has_aux=True)

        θ          = jnp.asarray(θ0)
        opt        = optax.adam(LR)
        opt_state  = opt.init(θ)

        for it in range(1, ITER+1):
            batch_idx = rng.choice(len(trajs), BATCH_SIZE, replace=True)

            for _ in range(INNER_ITERS):
                tot_L = 0.0;  tot_g = onp.zeros_like(θ0)

                for idx in batch_idx:
                    S, A, (gx,gy) = S_ls[idx], A_ls[idx], G_ls[idx]

                    w_vec, q_vec = _split_theta(θ)
                    cT.w, cT.gx, cT.gy = w_vec, gx, gy
                    cQ.q, cQ.gx, cQ.gy = q_vec, gx, gy

                    prev_Ps = prev_α = None
                    for x, a in zip(S, A):
                        solver.run(jnp.asarray(x)) if prev_Ps is None else \
                            solver.run(jnp.asarray(x),
                                       Ps_warm=prev_Ps, alphas_warm=prev_α)
                        xs, us      = solver._best_operating_point
                        prev_Ps     = solver._Ps
                        prev_α      = solver._alphas

                        u_h         = jnp.array([a[1], a[0]])
                        (sur, mse), g = value_grad(θ, xs, us, u_h, gx, gy)
                        tot_L      += float(mse); tot_g += onp.asarray(g)

                g = tot_g / (BATCH_SIZE * len(S))
                L = tot_L / (BATCH_SIZE * len(S))

                updates, opt_state = opt.update(jnp.asarray(g), opt_state, θ)
                θ = optax.apply_updates(θ, updates)

                if ENSURE_POS: θ = jnp.maximum(θ, 1e-9)

                # refresh cached solver so new weights are seen
                _SOLVER_CACHE.pop(H, None)
                solver, cT, cQ = _make_solver(H)

            if it % PRINT_EVERY == 0:
                print(f" it {it:3d}  loss {L:.6f}")
                print("    θ =", onp.asarray(θ).tolist())
                print("    g =", g.tolist())

        if L < best[0]:
            best = (L, θ, H)

    return best[1], best[2], best[0]


# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    θ0 = onp.array( [1e+3]*8 + [1.0], dtype=onp.float32 )   # λτ start at 1
    for pkl in args.datasets:
        print(f"\n===== {pkl} =====")
        trajs = pickle.load(open(pkl, "rb"))
        θ, H, L = train_dataset(trajs, θ0)
        print(f"Best H={H}, loss={L:.6f}")
        out = os.path.splitext(pkl)[0] + "_learned.npz"
        onp.savez(out, theta=onp.asarray(θ), H=int(H), loss=L)
        print("Saved →", out)
        if args.plot:
            validate(trajs, θ, H)


if __name__ == "__main__":
    main()
