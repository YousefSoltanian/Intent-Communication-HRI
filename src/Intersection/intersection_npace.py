#!/usr/bin/env python
"""
NPACE-style controller for the Intersection game (optimized with JAX)
---------------------------------------------------------------------

Robot (player-1, a2) is uncertain over human intent θ_h ∈ {1..5}.
Human (player-0, a1) is modeled as a Q-MDP solver that is uncertain over
the robot's intent θ_r ∈ {1..5}, with its own Bayesian update over θ_r.

This version keeps the exact logic of your original NPACE while using
vectorized JAX operations for the Bayesian updates and Q-MDP mixing.
All hyper-parameters, horizons, limits, and initial conditions remain
unchanged from your original setup.
"""

from __future__ import annotations
import math
from typing import Dict, Tuple, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

# ---- iLQGame imports ----
from iLQGame.cost        import UncontrolledIntersectionPlayerCost
from iLQGame.player_cost import PlayerCost
from iLQGame.ilq_solver  import ILQSolver
from iLQGame.constraint  import BoxConstraint
from iLQGame.multiplayer_dynamical_system import UncontrolledIntersection2PlayerSystem

# JAX helpers
@jax.jit
def _log_gauss_scalar(u: jnp.ndarray, mu: jnp.ndarray, s2: jnp.ndarray) -> jnp.ndarray:
    s2 = jnp.maximum(s2, 1e-9)
    return -0.5 * (u - mu) ** 2 / s2 - 0.5 * (jnp.log(2.0 * jnp.pi) + jnp.log(s2))

@jax.jit
def _curv(B: jnp.ndarray, Z1: jnp.ndarray, R0: jnp.ndarray) -> jnp.ndarray:
    return (R0 + B.T @ Z1 @ B).squeeze()

# Warm-startable ILQ solver for a fixed (θ_h, θ_r) pair
class _Solver2P_NPACE:
    def __init__(
        self,
        *,
        theta_h: float,
        theta_r: float,
        dt: float,
        horizon: int,
        effort_w: float,
        b_pen: float,
        gamma: float,
        mu: float,
        v_nom: float,
        R: float,
        W: float,
        L: float,
        acc_low: float,
        acc_high: float,
        max_iter: int = 25,
        verbose: bool = False,
    ):
        self.theta_h = float(theta_h)
        self.theta_r = float(theta_r)

        dyn = UncontrolledIntersection2PlayerSystem(T=float(dt))

        pc_h = PlayerCost()
        pc_r = PlayerCost()
        pc_h.add_cost(
            UncontrolledIntersectionPlayerCost(
                player_index=0, theta_self=self.theta_h, horizon=horizon,
                effort_weight=effort_w, b=b_pen, gamma=gamma, mu=mu,
                v_nom=v_nom, R=R, W=W, L=L
            ),
            arg="x", weight=1.0
        )
        pc_r.add_cost(
            UncontrolledIntersectionPlayerCost(
                player_index=1, theta_self=self.theta_r, horizon=horizon,
                effort_weight=effort_w, b=b_pen, gamma=gamma, mu=mu,
                v_nom=v_nom, R=R, W=W, L=L
            ),
            arg="x", weight=1.0
        )

        zeros_P = jnp.zeros((1, 4, horizon), dtype=jnp.float32)
        zeros_a = jnp.zeros((1,     horizon), dtype=jnp.float32)

        self.solver = ILQSolver(
            dyn,
            [pc_h, pc_r],
            [zeros_P, zeros_P],
            [zeros_a, zeros_a],
            max_iter=max_iter,
            u_constraints=[BoxConstraint(acc_low, acc_high),
                           BoxConstraint(acc_low, acc_high)],
            verbose=verbose
        )

        self._Ps = None
        self._a  = None

    def run(self, x: jnp.ndarray):
        self.solver.run(x, Ps_warm=self._Ps, alphas_warm=self._a)
        self._Ps, self._a = self.solver._Ps, self.solver._alphas

    def xs_nom(self):
        return self.solver._best_operating_point[0]  # (nx,N)

    def nom(self, pl: int) -> float:
        return float(self.solver._best_operating_point[1][pl][0, 0])

    def B_R(self, pl: int):
        xs, us = self.solver._best_operating_point
        _, Bs  = self.solver._linearize_dynamics(xs, us)
        _, _, _, H = self.solver._quadraticize_costs(xs, us)
        return Bs[pl][:, :, 0], H[pl][:, :, 0]

    def Zz_next(self, pl: int):
        Zs, zs = self.solver._best_op[4], self.solver._best_op[5]
        return Zs[pl][:, :, 1], zs[pl][:, 1]

# NPACE controller with JAX-accelerated mixing
class IntersectionNPACE:
    def __init__(
        self,
        *,
        theta_robot_true: int,
        intents: Tuple[int, ...] = (1, 2, 3, 4, 5),

        # time / geometry / costs (aligned with your intersection setup)
        dt: float = 0.1,
        horizon: int = 30,
        effort_w: float = 100.0,
        b_pen: float = 1e4,
        gamma: float = 0.5,
        mu: float = 1e-6,
        v_nom: float = 18.0,
        R: float = 70.0,
        W: float = 1.5,
        L: float = 3.0,

        # control limits
        acc_low: float = -50.0,
        acc_high: float = 50.0,

        # Bayes tuning
        beta_state: float = 1.0,
        rho_forget: float = 0.00,

        # state noise diag for human Bayes (x=[d1,v1,d2,v2])
        sigma2_state: Tuple[float, float, float, float] = (25000.0**1, 4000.0**1, 36.0**1, 1.0**1),

        max_iter: int = 25,
        verbose: bool = False,
    ):
        # Intent indexing
        self._intents = tuple(int(i) for i in intents)
        self._idx_of  = {th: i for i, th in enumerate(self._intents)}
        self._th_of   = {i: th for i, th in enumerate(self._intents)}

        # Robot's true intent index
        self._ir_true = self._idx_of[int(theta_robot_true)]

        # Basic constants
        self._effort  = float(effort_w)
        self._acc_lo, self._acc_hi = float(acc_low), float(acc_high)

        # Robot's belief over human θ_h
        nH = len(self._intents)
        self._b_h = np.ones(nH, dtype=np.float64) / nH

        # Modeled human beliefs over robot θ_r for each θ_h
        nR = len(self._intents)
        uni = np.ones(nR, dtype=np.float64) / nR
        self._q_r: Dict[int, np.ndarray] = {ih: uni.copy() for ih in range(nH)}

        # JAX state noise diag inverse & constants for Bayes over θ_r
        self._inv_sigma2 = jnp.asarray([1.0 / s for s in sigma2_state], dtype=jnp.float32)
        self._lgv_const  = float(-0.5 * (np.sum(np.log(sigma2_state)) + 4 * math.log(2.0 * math.pi)))
        self._beta_state = float(beta_state)
        self._rho_forget = float(rho_forget)

        # Parameters for solver construction
        self._params = dict(
            dt=float(dt), horizon=int(horizon),
            effort_w=float(effort_w), b_pen=float(b_pen), gamma=float(gamma),
            mu=float(mu), v_nom=float(v_nom), R=float(R), W=float(W), L=float(L),
            acc_low=float(acc_low), acc_high=float(acc_high),
            max_iter=int(max_iter), verbose=bool(verbose)
        )

        # Solver bank for each (θ_h, θ_r)
        self._solv: Dict[Tuple[int, int], _Solver2P_NPACE] = {}
        self._build_solvers()

        # Caches for Bayes updates and predictions
        self._pred_ctrl = None  # dict with μ_a1 and s2_a1 per θ_h
        self._x_cache   = None  # cached nominal next states shape (nH, nR, nx)

    def _build_solvers(self):
        p = self._params
        for ih, th_h in enumerate(self._intents):
            for ir, th_r in enumerate(self._intents):
                self._solv[(ih, ir)] = _Solver2P_NPACE(
                    theta_h=th_h, theta_r=th_r,
                    dt=p["dt"], horizon=p["horizon"],
                    effort_w=p["effort_w"], b_pen=p["b_pen"], gamma=p["gamma"],
                    mu=p["mu"], v_nom=p["v_nom"],
                    R=p["R"], W=p["W"], L=p["L"],
                    acc_low=p["acc_low"], acc_high=p["acc_high"],
                    max_iter=p["max_iter"], verbose=p["verbose"]
                )

    # Expose beliefs for inspection
    @property
    def robot_belief_over_human(self) -> Dict[int, float]:
        return {self._th_of[ih]: float(self._b_h[ih]) for ih in range(len(self._intents))}

    @property
    def modeled_human_beliefs(self) -> Dict[int, Dict[int, float]]:
        return {
            self._th_of[ih]: {self._th_of[ir]: float(self._q_r[ih][ir]) for ir in range(len(self._intents))}
            for ih in range(len(self._intents))
        }

    # Main NPACE update
    def compute_action(
        self,
        *,
        obs: np.ndarray,
        a1_observed: float
    ) -> float:
        x = jnp.asarray(obs, dtype=jnp.float32)
        nH, nR = len(self._intents), len(self._intents)
        ir_true = self._ir_true

        # (A) Human's Bayes over robot θ_r based on state x
        if self._x_cache is not None:
            x_cache = jnp.asarray(self._x_cache)        # shape (nH, nR, nx)
            errs = x[None, None, :] - x_cache           # (nH, nR, nx)
            logp = -0.5 * jnp.sum(errs * errs * self._inv_sigma2, axis=2)
            logp = self._beta_state * logp + self._lgv_const
            # Softmax over θ_r axis
            #exp_p = jax.nn.softmax(logp, axis=1)        # (nH, nR)
            # Apply forgetting factor toward uniform
            for ih in range(nH):
                prior = jnp.asarray(self._q_r[ih])
                log_post = jnp.log(prior + 1e-30) + logp[ih]
                q_new = jax.nn.softmax(log_post)
                self._q_r[ih] = (
                    (1.0 - self._rho_forget) * np.array(q_new, dtype=np.float64)
                    + self._rho_forget * (np.ones(nR) / nR)
                )

            

        # (B) Robot's Bayes over human θ_h using observed a1
        if self._pred_ctrl is not None:
            mu_arr = jnp.asarray(self._pred_ctrl["mu_a1"])  # (nH,)
            s2_arr = jnp.asarray(self._pred_ctrl["s2_a1"])  # (nH,)
            log_like = _log_gauss_scalar(jnp.asarray(a1_observed, jnp.float32), mu_arr, s2_arr)
            log_like = jnp.clip(log_like, -10.0, 10.0)
            prior_j  = jnp.asarray(self._b_h)
            logw = jnp.log(prior_j + 1e-30) + log_like
            self._b_h = np.array(jax.nn.softmax(logw), dtype=np.float64)
            # NEW addedition: clip extreme beliefs
            #eps = 1e-3
            #b = jnp.clip(jnp.asarray(self._b_h), eps, 1.0 - eps)
            #self._b_h = np.array(b / b.sum(), dtype=np.float64)

        # (C) Solve all (θ_h, θ_r) games (warm-started)
        for ih in range(nH):
            for ir in range(nR):
                self._solv[(ih, ir)].run(x)

        # (D) Human Q-MDP mixing over θ_r for each θ_h
        S1_mat = np.zeros((nH, nR), dtype=np.float64)
        l1_mat = np.zeros((nH, nR), dtype=np.float64)
        u1_mat = np.zeros((nH, nR), dtype=np.float64)
        # Gather curvatures, linear terms, and nominal controls
        for ih in range(nH):
            for ir in range(nR):
                s = self._solv[(ih, ir)]
                B1, _  = s.B_R(pl=0)
                Z1, z1 = s.Zz_next(pl=0)
                S1_mat[ih, ir] = float((B1.T @ Z1 @ B1).squeeze())
                l1_mat[ih, ir] = float((B1.T @ z1).squeeze())
                u1_mat[ih, ir] = s.nom(pl=0)
        # Convert to JAX arrays for vectorized Q-MDP
        S1_j = jnp.asarray(S1_mat)
        l1_j = jnp.asarray(l1_mat)
        u1_j = jnp.asarray(u1_mat)
        q_r_j= jnp.asarray(np.stack([self._q_r[ih] for ih in range(nH)]))
        # Human control per θ_h (vectorized)
        num_h = jnp.sum(q_r_j * (S1_j * u1_j - l1_j), axis=1)
        den_h = 2.0 * self._effort + jnp.sum(q_r_j * S1_j, axis=1)
        a1_Q_j= num_h / jnp.maximum(den_h, 1e-9)
        a1_Q  = np.array(a1_Q_j, dtype=np.float64)
        a1_nom_true = u1_mat[:, ir_true]  # (nH,)

        # (E) Robot Q-MDP mixing over θ_h with coupling to Δa1
        S2_arr = np.zeros(nH, dtype=np.float64)
        bias_arr = np.zeros(nH, dtype=np.float64)
        for ih in range(nH):
            s = self._solv[(ih, ir_true)]
            B2, _  = s.B_R(pl=1)
            Z2, z2 = s.Zz_next(pl=1)
            B1_true, _ = s.B_R(pl=0)
            delta_a1 = float(a1_Q[ih] - a1_nom_true[ih])
            l2_adj = float((B2.T @ z2).squeeze()) + 0.5*float((B2.T @ Z2 @ B1_true + B1_true.T @ Z2 @ B2).squeeze()) * delta_a1
            S2_val = float((B2.T @ Z2 @ B2).squeeze())
            S2_arr[ih] = S2_val
            bias_arr[ih] = S2_val * s.nom(pl=1) - l2_adj
        # Vectorized final mixing for a2
        bias_j = jnp.asarray(bias_arr)
        S2_j   = jnp.asarray(S2_arr)
        b_h_j  = jnp.asarray(self._b_h)
        num_R  = jnp.sum(b_h_j * bias_j)
        den_R  = 2.0 * self._effort + jnp.sum(b_h_j * S2_j)
        a2_cmd = float(jnp.clip(num_R / jnp.maximum(den_R, 1e-9), self._acc_lo, self._acc_hi))

        # (F) Update caches (μ/σ² for a1, and next-state nominal x)
        mu_a1 = a1_Q.copy()
        s2_a1 = np.zeros(nH, dtype=np.float64)
        for ih in range(nH):
            S1_true = S1_mat[ih, ir_true] + 2 * self._effort
            #s2_a1[ih] = jnp.clip(1.0 / max(S1_true, 1e-9), 10, 1e20)
            s2 = 1.0 / max(S1_true, 1e-2)
            s2_a1[ih] = float(np.clip(s2, 1e-2, 1e3))
        self._pred_ctrl = {"mu_a1": mu_a1, "s2_a1": s2_a1}

        # Cache next-step nominal states for state Bayes
        nx = 4
        x_cache = np.zeros((nH, nR, nx), dtype=np.float32)
        for ih in range(nH):
            for ir in range(nR):
                x_cache[ih, ir, :] = np.asarray(self._solv[(ih, ir)].xs_nom()[:, 1], dtype=np.float32)
        self._x_cache = x_cache

        return a2_cmd
