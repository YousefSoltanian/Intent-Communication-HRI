#!/usr/bin/env python
# NPACE – Robot uncertain over y; human uncertain over x
# Players: 0 = F_h , 1 = ΔF_r (robot) , 2 = τ_h
# ------------------------------------------------------------------

from __future__ import annotations
import os, sys, math, numpy as np, jax
import jax.numpy as jnp

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from iLQGame.cost        import ThrustPlayerParamCost, TorquePlayerParamCost
from iLQGame.player_cost import PlayerCost
from iLQGame.ilq_solver  import ILQSolver
from iLQGame.constraint  import BoxConstraint
from iLQGame.multiplayer_dynamical_system import LunarLander3PlayerSystem

# ───────── constants (unchanged) ───────────────────────────────────
DT, HORIZON   = 1/60, 60
THR_LIMIT     = 2000.0
DELTA_LIMIT   = 5000.0
TORQUE_LIMIT  = 1000.0

GX            = (400., 800.)
GY            = (250., 650.)
COM_OFFSET    = 5.0

T_EFF, TAU_EFF = 0.01, 9.0
W_TRUE = jnp.array([968., 962., 1036., 1035.,  T_EFF])
Q_TRUE = jnp.array([971., 972., 1037.,  967., TAU_EFF])

λ_F = float(W_TRUE[4])
λ_R = float(W_TRUE[4])
λ_τ = float(Q_TRUE[4])

LOG_2PI = math.log(2 * math.pi)
BETA_X  = 0.9
RHO_FORGET = 0.01

SIGMA2_STATE = jnp.array([50.0**2, 50.0**2, 25.0**2, 25.0**2,
                          0.3**2, 0.3**2], jnp.float32)
_LOG_DET_SIGMA = float(jnp.sum(jnp.log(SIGMA2_STATE)))

# ══════════ fast, jit-compiled helpers ═════════════════════════════
@jax.jit
def _log_gauss_scalar_jit(x, mu, s2):
    s2 = jnp.maximum(s2, 1e-9)
    return -0.5 * (x - mu) ** 2 / s2 - 0.5 * (jnp.log(2 * jnp.pi) + jnp.log(s2))

@jax.jit
def _log_gauss_vec_jit(err):
    return -0.5 * jnp.sum(err ** 2 / SIGMA2_STATE)

@jax.jit
def _curv_jit(B, Z1, R0):
    return (R0 + (B.T @ Z1 @ B)).squeeze()

# ═════════ warm-startable ILQ wrapper ══════════════════════════════
class _Solver3P:
    def __init__(self, gx: float, gy: float):
        dyn    = LunarLander3PlayerSystem(T=DT)
        target = (gx, gy + COM_OFFSET)

        pc_h = PlayerCost(); pc_r = PlayerCost(); pc_tau = PlayerCost()
        for pc in (pc_h, pc_r):
            pc.add_cost(ThrustPlayerParamCost(W_TRUE, target,
                                              horizon=HORIZON), arg="x")
        pc_tau.add_cost(TorquePlayerParamCost(Q_TRUE, target,
                                              horizon=HORIZON), arg="x")

        z = jnp.zeros
        P0 = z((1,6,HORIZON)); a0 = z((1,HORIZON))

        self.solver = ILQSolver(
            dyn,
            [pc_h, pc_r, pc_tau],
            [P0, P0, P0], [a0, a0, a0], max_iter=10,
            u_constraints=[
                BoxConstraint(-THR_LIMIT,   THR_LIMIT),
                BoxConstraint(-DELTA_LIMIT, DELTA_LIMIT),
                BoxConstraint(-TORQUE_LIMIT,TORQUE_LIMIT)],
            verbose=False)
        self._Ps = self._α = None

    def run(self, x):
        self.solver.run(x, Ps_warm=self._Ps, alphas_warm=self._α)
        self._Ps, self._α = self.solver._Ps, self.solver._alphas

    def xs_nom(self):                 return self.solver._best_operating_point[0]
    def nom(self, pl:int) -> float:   return float(self.solver._best_operating_point[1][pl][0,0])
    def Zz_next(self, pl:int):        Zs, zs = self.solver._best_op[4], self.solver._best_op[5]; return Zs[pl][:,:,1], zs[pl][:,1]
    def B_R(self, pl:int):            xs, us = self.solver._best_operating_point; _, Bs = self.solver._linearize_dynamics(xs, us); _,_,_,H = self.solver._quadraticize_costs(xs, us); return Bs[pl][:,:,0], H[pl][:,:,0]

# ═════════ NPACE main class ════════════════════════════════════════
class NPACE:
    def __init__(self, *, goal_x: float):
        self.ix_true = 0 if abs(goal_x-GX[0]) < 1e-6 else 1
        self.solv = {(ix,iy): _Solver3P(GX[ix], GY[iy])
                     for ix in (0,1) for iy in (0,1)}

        self.b_low = 0.5
        self.qx    = {0:0.5, 1:0.5}

        self._pred_ctrl   = None
        self._state_cache = None
        self._lgv_const   = -0.5*(_LOG_DET_SIGMA + 6*LOG_2PI)

    # ---------- public getters -------------------------------------
    @property
    def belief_y(self): return self.b_low
    @property
    def belief_x(self): return (self.qx[0], self.qx[1])

    # ---------- compute action -------------------------------------
    def compute_action(self,
                       obs       : np.ndarray,
                       human_tau : float,
                       human_thr : float) -> float:

        x   = jnp.asarray(obs, jnp.float32)
        τh  = -human_tau
        Fh  =  human_thr

        # (A) ----- human Bayes over x (vectorised) ------------------
        if self._state_cache is not None:
            errs = jnp.stack([x - self._state_cache[0][0],
                              x - self._state_cache[0][1],
                              x - self._state_cache[1][0],
                              x - self._state_cache[1][1]]).reshape(2,2,-1)
            logp = jax.vmap(jax.vmap(_log_gauss_vec_jit))(errs)*BETA_X + self._lgv_const
            probs = jnp.exp(logp - jnp.max(logp, axis=1, keepdims=True))
            post  = probs[:,0] / (jnp.sum(probs, axis=1) + 1e-12)
            self.qx[0] = float((1-RHO_FORGET)*post[0] + RHO_FORGET*0.5)
            self.qx[1] = float((1-RHO_FORGET)*post[1] + RHO_FORGET*0.5)

        # (B) ----- robot Bayes over y (vectorised) ------------------
        if self._pred_ctrl is not None:
            μτ,μF = self._pred_ctrl["τh"], self._pred_ctrl["Fh"]
            s2τ,s2F = self._pred_ctrl["s2τ"], self._pred_ctrl["s2F"]

            log_like = jnp.stack([
                _log_gauss_scalar_jit(τh, μτ[0], s2τ[0]) +
                _log_gauss_scalar_jit(Fh, μF[0], s2F[0]),
                _log_gauss_scalar_jit(τh, μτ[1], s2τ[1]) +
                _log_gauss_scalar_jit(Fh, μF[1], s2F[1])])
            w = jnp.exp(log_like - jnp.max(log_like))
            self.b_low = float((self.b_low*w[0]) /
                               (self.b_low*w[0] + (1-self.b_low)*w[1] + 1e-12))

        # (C) ----- solve four ILQ games -----------------------------
        for s in self.solv.values():
            s.run(x)

        # (D) ----- human Q-MDP (unchanged) --------------------------
        FQ, τQ, SF, Sτ = {}, {}, {}, {}
        for iy in (0,1):
            qL = self.qx[iy]

            Bτ,Rτ0 = self.solv[0,iy].B_R(2)
            Zτ_L,_= self.solv[0,iy].Zz_next(2); Zτ_R,_= self.solv[1,iy].Zz_next(2)
            Sτ[iy]=(float(_curv_jit(Bτ,Zτ_L,Rτ0)), float(_curv_jit(Bτ,Zτ_R,Rτ0)))
            τ_nom_L, τ_nom_R = self.solv[0,iy].nom(2), self.solv[1,iy].nom(2)
            τQ[iy] = -( qL*Sτ[iy][0]*τ_nom_L + (1-qL)*Sτ[iy][1]*τ_nom_R ) / \
                     ( 2*λ_τ + qL*Sτ[iy][0] + (1-qL)*Sτ[iy][1] )

            BF,RF0 = self.solv[0,iy].B_R(0)
            ZF_L,_= self.solv[0,iy].Zz_next(0); ZF_R,_= self.solv[1,iy].Zz_next(0)
            SF[iy]=(float(_curv_jit(BF,ZF_L,RF0)), float(_curv_jit(BF,ZF_R,RF0)))
            F_nom_L, F_nom_R = self.solv[0,iy].nom(0), self.solv[1,iy].nom(0)
            FQ[iy] = -( qL*SF[iy][0]*F_nom_L + (1-qL)*SF[iy][1]*F_nom_R ) / \
                     ( 2*λ_F + qL*SF[iy][0] + (1-qL)*SF[iy][1] )

        # (E) ----- robot ΔF_r Q-MDP across y (unchanged) ------------
        bias,SΔ={},{}
        for iy in (0,1):
            sol    = self.solv[self.ix_true, iy]
            BΔ,RΔ0 = sol.B_R(1)
            ZΔ,zΔ  = sol.Zz_next(1)
            BF,_   = sol.B_R(0);   Bτ,_ = sol.B_R(2)
            dF = FQ[iy] - sol.nom(0); dτ = τQ[iy] - sol.nom(2)
            lΔ = float((BΔ.T@zΔ).squeeze()) \
               + float((BΔ.T@ZΔ@BF).squeeze())*dF \
               + float((BΔ.T@ZΔ@Bτ).squeeze())*dτ
            SΔ[iy]   = float(_curv_jit(BΔ, ZΔ, RΔ0))
            bias[iy] = SΔ[iy]*sol.nom(1) - lΔ

        num   = self.b_low*bias[0] + (1-self.b_low)*bias[1]
        denom = 2*λ_R + self.b_low*SΔ[0] + (1-self.b_low)*SΔ[1]
        ΔF_cmd = float(np.clip(num/denom, -DELTA_LIMIT, DELTA_LIMIT))

        # (F) ----- caches for next step (unchanged) -----------------
        self._pred_ctrl = {
            "τh" : np.array([τQ[0], τQ[1]]),
            "Fh" : np.array([FQ[0], FQ[1]]),
            "s2τ": np.array([1.0 / Sτ[0][self.ix_true],
                             1.0 / Sτ[1][self.ix_true]]),
            "s2F": np.array([1.0 / SF[0][self.ix_true],
                             1.0 / SF[1][self.ix_true]])
        }

        self._state_cache = {
            iy: (self.solv[0,iy].xs_nom()[:,1],
                 self.solv[1,iy].xs_nom()[:,1]) for iy in (0,1)}

        return ΔF_cmd
