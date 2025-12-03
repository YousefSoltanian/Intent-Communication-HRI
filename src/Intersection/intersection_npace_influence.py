#!/usr/bin/env python
"""
Intersection NPACE-Influence
----------------------------
Same as your corrected Intersection NPACE, plus a teaching term that
nudges the modeled human belief toward the robot's true intent by shaping
the next state. All other logic is identical to your NPACE.
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

ddt=0.1
Dyn = UncontrolledIntersection2PlayerSystem(T=float(ddt))
# ---------- JAX helpers ----------
@jax.jit
def _log_gauss_scalar(u: jnp.ndarray, mu: jnp.ndarray, s2: jnp.ndarray) -> jnp.ndarray:
    #s2 = jnp.maximum(s2, 1e-19)
    return -0.5 * (u - mu) ** 2 / s2 - 0.5 * (jnp.log(2.0 * jnp.pi) + jnp.log(s2))

@jax.jit
def _curv(B: jnp.ndarray, Z1: jnp.ndarray, R0: jnp.ndarray) -> jnp.ndarray:
    return (R0 + B.T @ Z1 @ B).squeeze()

@jax.jit
def _teach_gain(B2: jnp.ndarray, delta_mu: jnp.ndarray,
                inv_sigma2: jnp.ndarray) -> jnp.ndarray:
    # B2: (nx,1) Jacobian ∂x/∂a2 at k=0
    # delta_mu: (nx,)   = x_next(true) - sum_r q_r * x_next(r)
    # inv_sigma2: (nx,) diagonal inverse covariance used in state-Bayes
    return (B2.T @ (delta_mu * inv_sigma2)).squeeze() #* dt

# ---------- warm-startable ILQ solver (unchanged from NPACE) ----------
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

# ---------- NPACE-Influence (preserves your corrected NPACE) ----------
class IntersectionNPACEInfluence:
    def __init__(
        self,
        *,
        theta_robot_true: int,
        #intents: Tuple[int, ...] = (1, 2, 3, 4, 5),
        intents: Tuple[int, ...] = (1, 3, 5), # narrowed for faster demo
        # time / geometry / costs (same as your corrected NPACE)
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

        # Bayes tuning (same)
        beta_state: float = 1.,
        rho_forget: float = 0.00,

        # state noise diag for human Bayes (x=[d1,v1,d2,v2])
        sigma2_state: Tuple[float, float, float, float] = (25.0**2, 4.0**2, 25.0**2, 4.0**2),
        # teaching strength
        gamma_teach: float = 0,

        max_iter: int = 25,
        verbose: bool = False,
    ):
        # intents & indices
        self._intents = tuple(int(i) for i in intents)
        self._idx_of  = {th: i for i, th in enumerate(self._intents)}
        self._th_of   = {i: th for i, th in enumerate(self._intents)}
        self._ir_true = self._idx_of[int(theta_robot_true)]

        # constants
        self._effort  = float(effort_w)
        self._acc_lo, self._acc_hi = float(acc_low), float(acc_high)
        self._dt = float(dt)
        self._gamma_teach = float(gamma_teach)

        # beliefs
        nH = len(self._intents)
        nR = len(self._intents)
        self._b_h = np.ones(nH, dtype=np.float64) / nH
        uni = np.ones(nR, dtype=np.float64) / nR
        self._q_r: Dict[int, np.ndarray] = {ih: uni.copy() for ih in range(nH)}

        # state-Bayes constants
        self._inv_sigma2 = jnp.asarray([1.0 / s for s in sigma2_state], dtype=jnp.float32)
        self._lgv_const  = float(-0.5 * (np.sum(np.log(sigma2_state)) + 4 * math.log(2.0 * math.pi)))
        self._beta_state = float(beta_state)
        self._rho_forget = float(rho_forget)

        # solver params & bank
        self._params = dict(
            dt=self._dt, horizon=int(horizon),
            effort_w=float(effort_w), b_pen=float(b_pen), gamma=float(gamma),
            mu=float(mu), v_nom=float(v_nom), R=float(R), W=float(W), L=float(L),
            acc_low=float(acc_low), acc_high=float(acc_high),
            max_iter=int(max_iter), verbose=bool(verbose)
        )
        self._solv: Dict[Tuple[int, int], _Solver2P_NPACE] = {}
        for ih, th_h in enumerate(self._intents):
            for ir, th_r in enumerate(self._intents):
                self._solv[(ih, ir)] = _Solver2P_NPACE(theta_h=th_h, theta_r=th_r, **self._params)

        # caches
        self._pred_ctrl = None  # {"mu_a1": (nH,), "s2_a1": (nH,)}
        self._x_cache   = None  # (nH, nR, nx)
        self._sigma2_a1_obs = 1.0**2

    # inspectors
    @property
    def robot_belief_over_human(self) -> Dict[int, float]:
        return {self._th_of[ih]: float(self._b_h[ih]) for ih in range(len(self._intents))}

    @property
    def modeled_human_beliefs(self) -> Dict[int, Dict[int, float]]:
        return { self._th_of[ih]: { self._th_of[ir]: float(self._q_r[ih][ir])
                                    for ir in range(len(self._intents)) }
                 for ih in range(len(self._intents)) }

    # main step
    def compute_action(self, *, obs: np.ndarray, a1_observed: float) -> float:
        x = jnp.asarray(obs, dtype=jnp.float32)
        nH = len(self._intents); nR = len(self._intents)
        ir_true = self._ir_true

        # (A) Human's Bayes over robot θ_r based on state x (with prior)
        if self._x_cache is not None:
            x_cache = jnp.asarray(self._x_cache)         # (nH, nR, nx)
            errs    = x[None, None, :] - x_cache         # (nH, nR, nx)
            logp    = -0.5 * jnp.sum(errs * errs * self._inv_sigma2, axis=2)
            logp    = self._beta_state * logp + self._lgv_const
            for ih in range(nH):
                prior = jnp.asarray(self._q_r[ih])        # (nR,)
                log_post = jnp.log(prior) + logp[ih]
                #new
                #log_post = log_post - jnp.max(log_post)  # for numerical stability
                q_new = jax.nn.softmax(log_post)
                self._q_r[ih] = (
                    (1.0 - self._rho_forget) * np.array(q_new, dtype=np.float64)
                    + self._rho_forget * (np.ones(nR) / nR)
                )
                #self._q_r[ih] = np.clip(self._q_r[ih], 1e-3, 1 - 1e-3)
                #self._q_r[ih] /= self._q_r[ih].sum()

        # (B) Robot's Bayes over human θ_h using observed a1 (unchanged)
        if self._pred_ctrl is not None:
            mu_arr = jnp.asarray(self._pred_ctrl["mu_a1"])  # (nH,)
            s2_arr = jnp.asarray(self._pred_ctrl["s2_a1"])  # (nH,)
            s2_arr = s2_arr + self._sigma2_a1_obs
            # Variance floor/ceiling to avoid over-confidence & numerical spikes
            #s2_min = 25.0   # try 10–100 depending on your action scale
            #s2_max = 1e6
            #s2_arr = jnp.clip(s2_arr, s2_min, s2_max)
            log_like = _log_gauss_scalar(jnp.asarray(a1_observed, jnp.float32), mu_arr, s2_arr)
            #log_like = jnp.clip(log_like, -10.0, 10.0)
            prior_j  = jnp.asarray(self._b_h)
            post = jnp.log(prior_j) + log_like
            #new
            #post = post - jnp.max(post)  # for numerical stability
            self._b_h = np.array(jax.nn.softmax(post), dtype=np.float64)
            #self._b_h = jnp.clip(self._b_h, 1e-3, 1 - 1e-3)
            #self._b_h /= self._b_h.sum()

        # (C) Solve all (θ_h, θ_r)
        for ih in range(nH):
            for ir in range(nR):
                self._solv[(ih, ir)].run(x)

        # (D) Human Q-MDP over θ_r (unchanged), also collect x_next for teaching/cache
        H1_mat = np.zeros((nH, nR), dtype=np.float64)
        S1_mat = np.zeros((nH, nR), dtype=np.float64)
        l1_mat = np.zeros((nH, nR), dtype=np.float64)
        u1_mat = np.zeros((nH, nR), dtype=np.float64)
        nx = 4
        x1 = np.zeros((nH, nR, nx), dtype=np.float32)

        for ih in range(nH):
            for ir in range(nR):
                s = self._solv[(ih, ir)]
                B1, R01  = s.B_R(pl=0)
                #print(B1)
                Z1, z1 = s.Zz_next(pl=0)
                H1_mat[ih, ir]  = float((B1.T @ Z1 @ B1).squeeze())
                S1_mat[ih, ir] = float((R01 + B1.T @ Z1 @ B1).squeeze())
                #print(B1.T @ Z1 @ B1)
                l1_mat[ih, ir] = float((B1.T @ z1).squeeze())
                u1_mat[ih, ir] = s.nom(pl=0)
                x1[ih, ir, :]  = np.asarray(s.xs_nom()[:, 1], dtype=np.float32)

        H1_j   = jnp.asarray(H1_mat)
        S1_j = jnp.asarray(S1_mat)
        l1_j = jnp.asarray(l1_mat)
        u1_j = jnp.asarray(u1_mat)
        q_r_j= jnp.asarray(np.stack([self._q_r[ih] for ih in range(nH)]))

        ###### IMPORTANT CHANGE HERE ######
        #num_h = jnp.sum(q_r_j * (H1_j * u1_j - l1_j), axis=1)
        num_h = jnp.sum(q_r_j * (S1_j * u1_j), axis=1)
        #den_h = 2.0 * self._effort + jnp.sum(q_r_j * S1_j, axis=1)
        den_h = jnp.sum(q_r_j * S1_j, axis=1)

        #a1_Q_j= num_h / jnp.maximum(den_h, 1e-9)
        a1_Q_j= num_h /den_h
        #a1_Q_j = jnp.clip(a1_Q_j, self._acc_lo, self._acc_hi)
        a1_Q  = np.array(a1_Q_j, dtype=np.float64)
        #a1_Q  = float(jnp.clip(a1_Q_j, self._acc_lo, self._acc_hi))
        a1_nom_true = u1_mat[:, ir_true]  # (nH,)

        # (E) Robot Q-MDP across θ_h with coupling to Δa1  +  TEACHING
        S2_arr  = np.zeros(nH, dtype=np.float64)
        bias_arr= np.zeros(nH, dtype=np.float64)
        for ih in range(nH):
            s      = self._solv[(ih, ir_true)]
            B2, R02  = s.B_R(pl=1)
            #print(B2)
            Z2, z2 = s.Zz_next(pl=1)
            B1_true, _ = s.B_R(pl=0)
            # human deviation (robot models human as QMDP, so use a1_Q from above)
            #delta_a1 = float(a1_Q[ih] - a1_nom_true[ih])
            #a1_Q[ih]  = float(jnp.clip(a1_Q[ih], self._acc_lo, self._acc_hi))
            delta_a1 = float(a1_Q[ih]- a1_nom_true[ih])

            # H2 and S2 (denominator)
            H2_num = float((B2.T @ Z2 @ B2).squeeze())
            S2_den = float((np.array(R02).squeeze() + H2_num)) # R0_2 + H2
            #print(B2)

            # linear part: ℓ + C δa1
            l2_lin  = float((B2.T @ z2).squeeze())             # ℓ = B2^T z2
            #C_cross = float((0.5*B2.T @ Z2 @ B1_true + 0.5*B1_true.T @ Z2 @ B2).squeeze())   # C = B2^T Z2 B1
            C_cross = 0.5*float((B2.T @ Z2 @ B1_true + B1_true.T @ Z2 @ B2).squeeze())
            l2_adj  = l2_lin + C_cross * delta_a1

            # optional teaching term (kept as in your code; sign consistent with a* = (… - γ τ)/S)
            B2_vec       = B2.squeeze()
            inv_sig_teach= self._beta_state * self._inv_sigma2
            kappa_teach  = float(jnp.sum((B2_vec ** 2) * inv_sig_teach))
            #a2_true      = float(self._solv[(ih, ir_true)].nom(pl=1))
            a2_true      = np.array([self._solv[(ih, ir_true)].nom(pl=1) for ir in range(nR)], dtype=float)
            a2_all       = np.array([self._solv[(ih, ir)].nom(pl=1) for ir in range(nR)], dtype=float)
            q_vec        = np.asarray(self._q_r[ih], dtype=float)
            #gap          = float(np.dot(q_vec, (a2_all - a2_true)/(a2_true + 1)))
            gap          = float(np.dot(q_vec, (a2_all - a2_true)))
            g_teach      = kappa_teach * gap#*(1-q_vec[ir_true])  # same proxy as you used

            # numerator (a_r affine): H2*u2_nom - (l2_adj + γ*g_teach)

            ###### IMPORTANT CHANGE HERE ######
            #bias_arr[ih] = H2_num * s.nom(pl=1) - (l2_adj + self._gamma_teach * g_teach)
            bias_arr[ih] = S2_den * s.nom(pl=1) - (C_cross * delta_a1 + self._gamma_teach * g_teach)
            S2_arr[ih]   = S2_den
            

        # final mixing
        bias_j = jnp.asarray(bias_arr)
        S2_j   = jnp.asarray(S2_arr)
        b_h_j  = jnp.asarray(self._b_h)

        num_R  = jnp.sum(b_h_j * bias_j)
        den_R  = jnp.sum(b_h_j * S2_j)
        #a2_cmd = float(jnp.clip(num_R / den_R, self._acc_lo, self._acc_hi))
        a2_cmd = float(num_R / den_R)
        #a2_cmd = float(num_R / den_R)

        # (F) caches (unchanged), reuse x1 for Bayes
        mu_a1 = a1_Q.copy()
        #mu_a1 = float(jnp.clip(a1_Q[:], self._acc_lo, self._acc_hi))
        s2_a1 = np.zeros(nH, dtype=np.float64)
        for ih in range(nH):
            #S1_true = S1_mat[ih, ir_true] + 2 * self._effort
            S1_true = float(np.dot(self._q_r[ih], S1_mat[ih, :])) 
            #s2_a1[ih] = jnp.clip(1.0 / max(S1_true, 1e-9), 100, 1e20)
            #s2_a1[ih] = 1.0 / (self._beta_state*S1_true) + 20
            #s2_a1[ih] = 1.0 / (1*S1_true) #+ 20
            #s2 = 1.0 / (1*S1_true, 1e-19)
            #s2_a1[ih] = float(np.clip(s2, 1e-2, 1e3))
            #s2_a1[ih] = 1.0 / max(S1_true, 1e-19) #+ 1000
            #s2_a1[ih] = 1.0 / max(S1_true, 1e-10) 
            s2_a1[ih] = 1.0 / (S1_true)  
            #print(S1_true)  
        self._pred_ctrl = {"mu_a1": mu_a1, "s2_a1": s2_a1}

        # (F) Cache predicted next states using human QMDP action (per θ_h)
        # and robot nominal action (per θ_r branch)
        nx = int(x.shape[0])  # or 4 if you prefer
        x_cache = np.zeros((nH, nR, nx), dtype=np.float32)
        for ih in range(nH):
            # human QMDP action for this human-intent branch
            #u1_pred = float(jnp.clip(a1_observed, self._acc_lo, self._acc_hi))
            #u1_pred = float(a1_observed)
            u1_pred = float(a1_Q[ih])
            for ir in range(nR):
                # robot nominal for this (ih, ir) branch
                u2_pred = float(self._solv[(ih, ir)].nom(pl=1))
                # one-step plant rollout
                x_next = Dyn.disc_time_dyn(
                    x,
                    [jnp.array([u1_pred], dtype=jnp.float32),
                    jnp.array([u2_pred], dtype=jnp.float32)]
                )
                #x_cache[ih, ir, :] = np.asarray(self._solv[(ih, ir)].xs_nom()[:, 1], dtype=np.float32)
                x_cache[ih, ir, :] = np.asarray(x_next, dtype=np.float32)
        self._x_cache = x_cache
                                                                                                                                              
        return a2_cmd
