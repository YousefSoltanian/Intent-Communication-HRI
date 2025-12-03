#!/usr/bin/env python
# NPACE-Influence – identical maths, fewer Python hops, faster JAX
# ------------------------------------------------------------------
# * No constants, equations, or limits have changed.
# * Only micro-optimisations:  JIT-compiled helper kernels and
#   vectorised likelihoods to trim Python overhead.
# ------------------------------------------------------------------

from __future__ import annotations
import os, sys, math, jax, jax.numpy as jnp, numpy as np

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
DELTA_LIMIT   = 4000.0
TORQUE_LIMIT  = 1000.0

GX,  GY       = (400., 800.), (250., 650.)
COM_OFFSET    = 5.0

T_EFF, TAU_EFF = 0.01, 9.0
W_TRUE = jnp.array([968., 962., 1036., 1035.,  T_EFF])
Q_TRUE = jnp.array([971., 972., 1037.,  967.,  TAU_EFF])

λ_F = float(W_TRUE[4]);  λ_R = float(W_TRUE[4]);  λ_τ = float(Q_TRUE[4])

LOG_2PI   = math.log(2*math.pi)
BETA_X    = 0.9
GAMMA_LL  = 1.0e6
RHO_FORGET= 0.01

SIGMA2_STATE = jnp.array([50.**2, 50.**2, 25.**2, 25.**2, 0.3**2, 0.3**2],
                         jnp.float32)
INV_SIGMA2_STATE = 1.0 / SIGMA2_STATE
LOG_NORM_STATE   = 0.5*(SIGMA2_STATE.size*LOG_2PI
                        + jnp.sum(jnp.log(SIGMA2_STATE)))

# ═════════ jit-compiled helper kernels (math unchanged) ════════════
@jax.jit
def _log_gauss_vec(err):                        # err shape (...,6)
    return -0.5*jnp.sum(err*err*INV_SIGMA2_STATE, axis=-1) - LOG_NORM_STATE

@jax.jit
def _curv(B, Z1, R0):                           # 1×6 • 6×6 • 6×1 → scalar
    return (R0 + B.T @ Z1 @ B).squeeze()

@jax.jit
def _teach_gain(BΔ, dX):                        # derivative wrt ΔF_r of log-LLR
    return (BΔ.T @ (dX*INV_SIGMA2_STATE)).squeeze() * DT

@jax.jit
def _ln_like_scalar(v, mu, s2):                 # scalar Gaussian log-pdf
    s2 = jnp.maximum(s2, 1e-9)
    return -0.5*((v-mu)**2)/s2 - 0.5*(jnp.log(s2)+LOG_2PI)

# ═════════ warm-startable ILQ wrapper (unchanged maths) ════════════
class _Solver3P:
    """One (x,y) branch of the 3-player ILQ game."""
    def __init__(self, gx: float, gy: float):
        dyn    = LunarLander3PlayerSystem(T=DT)
        target = (gx, gy + COM_OFFSET)

        pc_h = PlayerCost(); pc_r = PlayerCost(); pc_tau = PlayerCost()
        for pc in (pc_h, pc_r):
            pc.add_cost(ThrustPlayerParamCost(W_TRUE, target, horizon=HORIZON),
                         arg="x")
        pc_tau.add_cost(TorquePlayerParamCost(Q_TRUE, target, horizon=HORIZON),
                        arg="x")

        z = jnp.zeros
        P0, a0 = z((1,6,HORIZON)), z((1,HORIZON))

        self.solver = ILQSolver(
            dyn, [pc_h, pc_r, pc_tau],
            [P0,P0,P0], [a0,a0,a0], max_iter=10,
            u_constraints=[
                BoxConstraint(-THR_LIMIT,   THR_LIMIT),
                BoxConstraint(-DELTA_LIMIT, DELTA_LIMIT),
                BoxConstraint(-TORQUE_LIMIT,TORQUE_LIMIT)],
            verbose=False)
        self._Ps = self._α = None

    def run(self, x):                      # warm-start each step
        self.solver.run(x, Ps_warm=self._Ps, alphas_warm=self._α)
        self._Ps, self._α = self.solver._Ps, self.solver._alphas

    # ----- thin wrappers used by controller (unchanged) ------------
    def xs_nom(self):               return self.solver._best_operating_point[0]
    def nom(self, pl:int) -> float: return float(
                                    self.solver._best_operating_point[1][pl][0,0])
    def Zz_next(self, pl:int):
        Zs, zs = self.solver._best_op[4], self.solver._best_op[5]
        return Zs[pl][:,:,1], zs[pl][:,1]
    def B_R(self, pl:int):
        xs, us = self.solver._best_operating_point
        _, Bs  = self.solver._linearize_dynamics(xs, us)
        _,_,_,H= self.solver._quadraticize_costs(xs, us)
        return Bs[pl][:,:,0], H[pl][:,:,0]

# ═════════ NPACE-Influence main class (logic unchanged) ════════════
class NPACE_Influence:
    def __init__(self, *, goal_x: float):
        self.ix_true = 0 if abs(goal_x-GX[0]) < 1e-6 else 1
        self.solv    = {(ix,iy): _Solver3P(GX[ix], GY[iy])
                        for ix in (0,1) for iy in (0,1)}
        self.b_low   = 0.5
        self.qx      = jnp.array([0.5, 0.5])
        self._pred_ctrl   = None
        self._state_cache = None                       # shape (2,2,6)

    # public inspectors
    @property
    def belief_y(self): return self.b_low
    @property
    def belief_x(self): return tuple(self.qx)

    # main step
    def compute_action(self,
                       obs       : np.ndarray,
                       human_tau : float,
                       human_thr : float) -> float:

        x  = jnp.asarray(obs, jnp.float32)
        τh = -float(human_tau);  Fh = float(human_thr)

        # (1) human Bayes over x-goal --------------------------------
        if self._state_cache is not None:
            err   = x - self._state_cache            # (2,2,6) broadcasting
            logL  = _log_gauss_vec(err[:,0,:]) * BETA_X
            logR  = _log_gauss_vec(err[:,1,:]) * BETA_X
            pL    = self.qx * jnp.exp(logL)
            post  = pL / (pL + (1-self.qx)*jnp.exp(logR) + 1e-12)
            self.qx = (1-RHO_FORGET)*post + RHO_FORGET*0.5

        # (2) robot Bayes over y-goal --------------------------------
        if self._pred_ctrl is not None:
            μτ,s2τ = self._pred_ctrl["τh"],  self._pred_ctrl["s2τ"]
            μF,s2F = self._pred_ctrl["Fh"],  self._pred_ctrl["s2F"]

            log_like = jnp.stack([
                _ln_like_scalar(τh, μτ[0], s2τ[0]) +
                _ln_like_scalar(Fh, μF[0], s2F[0]),
                _ln_like_scalar(τh, μτ[1], s2τ[1]) +
                _ln_like_scalar(Fh, μF[1], s2F[1])])
            w = jnp.exp(log_like - jnp.max(log_like))
            self.b_low = float(self.b_low*w[0] /
                               (self.b_low*w[0] + (1-self.b_low)*w[1] + 1e-12))

        # (3) solve four ILQ games -----------------------------------
        for s in self.solv.values():
            s.run(x)

        # (4) human Q-MDP -------------------------------------------
        FQ, τQ   = np.zeros(2), np.zeros(2)
        SF,  Sτ  = np.zeros((2,2)), np.zeros((2,2))
        for iy in (0,1):
            qL = self.qx[iy]

            Bτ,Rτ0 = self.solv[0,iy].B_R(2)
            ZτL,_  = self.solv[0,iy].Zz_next(2);  ZτR,_ = self.solv[1,iy].Zz_next(2)
            Sτ[iy]=( _curv(Bτ,ZτL,Rτ0), _curv(Bτ,ZτR,Rτ0) )
            τ_nomL, τ_nomR = self.solv[0,iy].nom(2), self.solv[1,iy].nom(2)
            τQ[iy] = -(qL*Sτ[iy][0]*τ_nomL + (1-qL)*Sτ[iy][1]*τ_nomR) / \
                     (2*λ_τ + qL*Sτ[iy][0] + (1-qL)*Sτ[iy][1])

            BF,RF0 = self.solv[0,iy].B_R(0)
            ZFL,_  = self.solv[0,iy].Zz_next(0);  ZFR,_ = self.solv[1,iy].Zz_next(0)
            SF[iy]=( _curv(BF,ZFL,RF0), _curv(BF,ZFR,RF0) )
            F_nomL, F_nomR = self.solv[0,iy].nom(0), self.solv[1,iy].nom(0)
            FQ[iy] = -(qL*SF[iy][0]*F_nomL + (1-qL)*SF[iy][1]*F_nomR) / \
                     (2*λ_F + qL*SF[iy][0] + (1-qL)*SF[iy][1])

        # (5) robot ΔF_r Q-MDP + teaching ----------------------------
        bias, SΔ = np.zeros(2), np.zeros(2)
        for iy in (0,1):
            sol        = self.solv[self.ix_true, iy]
            BΔ,RΔ0     = sol.B_R(1)
            ZΔ,zΔ      = sol.Zz_next(1)
            BF,_       = sol.B_R(0);  Bτ,_ = sol.B_R(2)

            dF,dτ      = FQ[iy]-sol.nom(0), τQ[iy]-sol.nom(2)
            lΔ         = float((BΔ.T@zΔ).squeeze() +
                               (BΔ.T@ZΔ@BF).squeeze()*dF +
                               (BΔ.T@ZΔ@Bτ).squeeze()*dτ)
            SΔ[iy]     = _curv(BΔ, ZΔ, RΔ0)

            # teaching term (kernel unchanged, but jit-compiled)
            xL1 = self.solv[0,iy].xs_nom()[:,1]
            xR1 = self.solv[1,iy].xs_nom()[:,1]
            g   = float(_teach_gain(BΔ, xR1-xL1))
            gap = (1.0 if self.ix_true==0 else -1.0) #- self.qx[iy]

            dy  = float(abs(x[1]-GY[iy])); dx = float(abs(x[0]-GX[self.ix_true]))
            #fade = np.clip(np.exp(max(dx,dy)/50.0), 0., 1.)
            fade = 1

            bias[iy] = SΔ[iy]*sol.nom(1) - lΔ + fade*GAMMA_LL*gap*g

        ΔF_cmd = float(np.clip(
            (self.b_low*bias[0] + (1-self.b_low)*bias[1]) /
            (2*λ_R + self.b_low*SΔ[0] + (1-self.b_low)*SΔ[1]),
            -DELTA_LIMIT, DELTA_LIMIT))

        # (6) cache for next frame ----------------------------------
        self._pred_ctrl = {
            "τh":  τQ,
            "Fh":  FQ,
            "s2τ": np.array([1.0/Sτ[0][self.ix_true],
                             1.0/Sτ[1][self.ix_true]]),
            "s2F": np.array([1.0/SF[0][self.ix_true],
                             1.0/SF[1][self.ix_true]])
        }
        self._state_cache = jnp.stack(
            [jnp.stack([self.solv[0,iy].xs_nom()[:,1],
                        self.solv[1,iy].xs_nom()[:,1]])
             for iy in (0,1)])                         # (2,2,6)

        return ΔF_cmd
