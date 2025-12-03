#!/usr/bin/env python
"""
Blame-Me Δ-thrust controller – exact Q-MDP (robot uncertain about y-pad)

Information sources:
    • τ_h  (human torque)         – as before
    • F_h  (human thrust bias)    – NEW

Real-time: one ILQ solve per frame per hypothesis, warm-started.
"""

from __future__ import annotations
import os, sys, math
import jax.numpy as jnp
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from iLQGame.cost        import ThrustPlayerParamCost, TorquePlayerParamCost
from iLQGame.player_cost import PlayerCost
from iLQGame.ilq_solver  import ILQSolver
from iLQGame.constraint  import BoxConstraint
from iLQGame.multiplayer_dynamical_system import LunarLander3PlayerSystem

# ─── constants ──────────────────────────────────────────────────────
DT, HORIZON = 1/60, 60
DELTA_LIMIT = 5000.0
THR_LIMIT   = 2000.0
TORQUE_MAX  = 2000.0
GY          = (250., 650.)            # low / high
COM_OFFSET  = 5.0                    # px

T_EFF, TAU_EFF = 0.01, 9.0
W_TRUE = jnp.array([968., 962., 1036., 1035.,  T_EFF])
Q_TRUE = jnp.array([971., 972., 1037.,  967., TAU_EFF])

LOG_2PI = math.log(2*math.pi)

# ─── warm-startable solver per y-hypothesis ─────────────────────────
class _Solver3P:
    def __init__(self, gx: float, gy: float):
        dyn = LunarLander3PlayerSystem(T=DT)

        pc_h   = PlayerCost()
        pc_r   = PlayerCost()
        pc_tau = PlayerCost()

        target = (gx, gy + COM_OFFSET)
        pc_h  .add_cost(ThrustPlayerParamCost(1*W_TRUE, target, horizon=HORIZON),
                        arg="x")                  # weight 1 → informative
        pc_r  .add_cost(ThrustPlayerParamCost(1*W_TRUE, target, horizon=HORIZON),
                        arg="x")
        pc_tau.add_cost(TorquePlayerParamCost(Q_TRUE, target, horizon=HORIZON),
                        arg="x")

        z = jnp.zeros
        zeros_P = z((1,6,HORIZON))
        zeros_a = z((1,  HORIZON))

        self.solver = ILQSolver(
            dyn,
            [pc_h, pc_r, pc_tau],
            [zeros_P]*3,
            [zeros_a]*3,
            max_iter=5,
            u_constraints=[
                BoxConstraint(-THR_LIMIT, THR_LIMIT),   # F_h free
                BoxConstraint(-DELTA_LIMIT, DELTA_LIMIT),
                BoxConstraint(-TORQUE_MAX, TORQUE_MAX)],
            verbose=False)

        self._Ps = self._a = None

    # --------------- step ------------------------------------------
    def run(self, x):
        self.solver.run(x, Ps_warm=self._Ps, alphas_warm=self._a)
        self._Ps, self._a = self.solver._Ps, self.solver._alphas

    # --------------- nominal controls & linear coeffs --------------
    def nom(self, pl:int) -> float:
        return float(self.solver._best_operating_point[1][pl][0,0])

    def B_R(self, pl:int):
        xs, us = self.solver._best_operating_point
        _, Bs  = self.solver._linearize_dynamics(xs, us)
        _,_,_,H= self.solver._quadraticize_costs(xs, us)
        return Bs[pl][:,:,0], H[pl][:,:,0]          # B, R0

    def Zz_next(self, pl:int):
        Zs, zs = self.solver._best_op[4], self.solver._best_op[5]
        return Zs[pl][:,:,1], zs[pl][:,1]

# ─── main controller ────────────────────────────────────────────────
class BlameMeThrustController:
    def __init__(self, *, goal_x: float):
        self._low  = _Solver3P(goal_x, GY[0])
        self._high = _Solver3P(goal_x, GY[1])
        self._b_low = 0.5
        self._λ = float(W_TRUE[4])
        self._pred = None

    # ---------- helpers --------------------------------------------
    @staticmethod
    def _log_gauss(u, mu, sigma2):
        return -0.5*((u-mu)**2)/sigma2 - 0.5*(LOG_2PI + math.log(sigma2))

    @staticmethod
    def _curvature(B, Z1, R0):
        return float((R0 + B.T @ Z1 @ B).squeeze())

    # ---------- main API -------------------------------------------
    def compute_action(self,
                       obs:   np.ndarray,
                       human_tau: float,
                       human_thr: float) -> float:

        x = jnp.asarray(obs, jnp.float32)
        τ_h = -float(human_tau)            # sign flip (env uses -τ_h)
        F_h =  float(human_thr)

        # --- 0. Bayesian update -----------------------------------
        if self._pred is not None:
            μτ_lo, μτ_hi = self._pred["tau_lo"],  self._pred["tau_hi"]
            σ2τ_lo,σ2τ_hi= self._pred["s2τ_lo"], self._pred["s2τ_hi"]
            μF_lo, μF_hi = self._pred["F_lo"],   self._pred["F_hi"]
            σ2F_lo,σ2F_hi= self._pred["s2F_lo"], self._pred["s2F_hi"]

            log_lo = ( self._log_gauss(τ_h, μτ_lo, σ2τ_lo) +
                       self._log_gauss(F_h, μF_lo, σ2F_lo) )
            log_hi = ( self._log_gauss(τ_h, μτ_hi, σ2τ_hi) +
                       self._log_gauss(F_h, μF_hi, σ2F_hi) )

            p_lo = self._b_low * math.exp(log_lo)
            p_hi = (1-self._b_low) * math.exp(log_hi)
            self._b_low = p_lo / (p_lo + p_hi + 1e-12)

        # --- 1. ILQ solves ---------------------------------------
        self._low.run(x);  self._high.run(x)

        # --- 2. Q-MDP closed form ΔF_r ---------------------------
        BΔ_lo,_ = self._low.B_R(1)
        BΔ_hi,_ = self._high.B_R(1)
        ZΔ_lo,zΔ_lo = self._low.Zz_next(1)
        ZΔ_hi,zΔ_hi = self._high.Zz_next(1)

        S_lo = float((BΔ_lo.T @ ZΔ_lo @ BΔ_lo).squeeze())
        S_hi = float((BΔ_hi.T @ ZΔ_hi @ BΔ_hi).squeeze())
        l_lo = float((BΔ_lo.T @ zΔ_lo).squeeze())
        l_hi = float((BΔ_hi.T @ zΔ_hi).squeeze())

        bias_lo = S_lo * self._low.nom(1)  - l_lo
        bias_hi = S_hi * self._high.nom(1) - l_hi

        num   = self._b_low * bias_lo + (1-self._b_low) * bias_hi
        denom = 2*self._λ + self._b_low*S_lo + (1-self._b_low)*S_hi
        delta_cmd = np.clip(num / denom, -DELTA_LIMIT, DELTA_LIMIT)

        # --- 3. Predict next-frame statistics for τ and F_h -------
        #   τ
        Bτ_lo,Rτ0_lo = self._low.B_R(2)
        Bτ_hi,Rτ0_hi = self._high.B_R(2)
        Zτ_lo,_ = self._low.Zz_next(2)
        Zτ_hi,_ = self._high.Zz_next(2)
        s2τ_lo = 1.0 / self._curvature(Bτ_lo, Zτ_lo, Rτ0_lo)
        s2τ_hi = 1.0 / self._curvature(Bτ_hi, Zτ_hi, Rτ0_hi)
        #   F_h
        BF_lo,RF0_lo = self._low.B_R(0)
        BF_hi,RF0_hi = self._high.B_R(0)
        ZF_lo,_ = self._low.Zz_next(0)
        ZF_hi,_ = self._high.Zz_next(0)
        s2F_lo = 1.0 / self._curvature(BF_lo, ZF_lo, RF0_lo)
        s2F_hi = 1.0 / self._curvature(BF_hi, ZF_hi, RF0_hi)

        self._pred = dict(tau_lo = self._low.nom(2),
                          tau_hi = self._high.nom(2),
                          F_lo   = self._low.nom(0),
                          F_hi   = self._high.nom(0),
                          s2τ_lo = s2τ_lo, s2τ_hi = s2τ_hi,
                          s2F_lo = s2F_lo, s2F_hi = s2F_hi)
        
        w_conf = 1.0 - 4.0 * self._b_low* (1.0 - self._b_low)     # 0 at 0.5, 1 at 0/1
        alpha  = 1 / (1 + np.exp(-12 * (w_conf - 0.5)))                        # no proximity factor
        F_bias_est = (self._b_low*self._low.nom(0) + (1-self._b_low)*self._high.nom(0))
        bias_cmd  = alpha * (F_bias_est - human_thr)
        return delta_cmd #+ alpha*human_thr

    # ---------- inspection helper ---------------------------------
    @property
    def belief_y(self): return self._b_low
