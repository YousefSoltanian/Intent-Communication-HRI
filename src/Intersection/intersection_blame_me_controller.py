#!/usr/bin/env python
"""
Blame-Me acceleration controller – exact Q-MDP (opponent θ ∈ {1..5})

• Game: 2-player uncontrolled intersection, x = [d1, v1, d2, v2]
         ḋ = v, v̇ = a   (discrete step DT)
• Intent: our θ_self is provided by the simulator (can be changed online).
          Opponent θ ∈ {1,2,3,4,5} with Bayesian update from observed
          opponent acceleration a_opp.
• Real-time: one ILQ solve per opponent hypothesis per frame, warm-started.
• Q-MDP: closed-form action from value-curvature and linear terms,
         mixed by the current belief over opponent θ.

Parameters are aligned with your paper baseline:
  DT=0.1, H=30, γ=0.5, b=1e4, μ=1e-6, v_nom=18, R=70, W=1.5, L=3.0,
  accel limits: [-500, 1000], effort_weight=1.0
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import jax.numpy as jnp

from iLQGame.cost        import UncontrolledIntersectionPlayerCost
from iLQGame.player_cost import PlayerCost
from iLQGame.ilq_solver  import ILQSolver
from iLQGame.constraint  import BoxConstraint
from iLQGame.multiplayer_dynamical_system import UncontrolledIntersection2PlayerSystem


# ╭────────────────────────────────────────────────────────────────╮
# │                Internal warm-startable ILQ wrapper             │
# ╰────────────────────────────────────────────────────────────────╯
class _Solver2P:
    """
    One ILQ solver instance for a *fixed* intent pair (θ_self, θ_opp).
    Keeps its own warm-start (Ps, alphas) to avoid reinitializing each frame.
    """

    def __init__(
        self,
        *,
        theta_self: float,
        theta_opp: float,
        horizon: int,
        dt: float,
        # cost/geometry
        effort_w: float,
        b_pen: float,
        gamma: float,
        mu: float,
        v_nom: float,
        R: float,
        W: float,
        L: float,
        # control limits
        acc_low: float,
        acc_high: float,
        # which player we control in the outer controller (0 or 1)
        ctrl_index_self: int,
        max_iter: int = 25,
        verbose: bool = False,
    ):
        self.theta_self = float(theta_self)
        self.theta_opp  = float(theta_opp)
        self.horizon    = int(horizon)
        self.ctrl_i     = int(ctrl_index_self)  # which player we control

        dyn = UncontrolledIntersection2PlayerSystem(T=float(dt))

        # per-player costs (θ goes into "theta_self" for that player's σ-window)
        pc_1 = PlayerCost()
        pc_2 = PlayerCost()

        # Map θs to players depending on who "self" is:
        theta_p0 = self.theta_self if self.ctrl_i == 0 else self.theta_opp
        theta_p1 = self.theta_opp  if self.ctrl_i == 0 else self.theta_self

        pc_1.add_cost(
            UncontrolledIntersectionPlayerCost(
                player_index=0, theta_self=theta_p0, horizon=horizon,
                effort_weight=effort_w, b=b_pen, gamma=gamma,
                mu=mu, v_nom=v_nom, R=R, W=W, L=L
            ),
            arg="x", weight=1.0
        )
        pc_2.add_cost(
            UncontrolledIntersectionPlayerCost(
                player_index=1, theta_self=theta_p1, horizon=horizon,
                effort_weight=effort_w, b=b_pen, gamma=gamma,
                mu=mu, v_nom=v_nom, R=R, W=W, L=L
            ),
            arg="x", weight=1.0
        )

        acc_limit_1 = BoxConstraint(acc_low, acc_high)
        acc_limit_2 = BoxConstraint(acc_low, acc_high)

        zeros_P = jnp.zeros((1, 4, horizon), dtype=jnp.float32)
        zeros_a = jnp.zeros((1,     horizon), dtype=jnp.float32)

        self.solver = ILQSolver(
            dyn,
            [pc_1, pc_2],
            [zeros_P, zeros_P],
            [zeros_a, zeros_a],
            max_iter=max_iter,
            u_constraints=[acc_limit_1, acc_limit_2],
            verbose=verbose
        )

        self._Ps = None
        self._a  = None

    def run(self, x: jnp.ndarray):
        """One warm-started ILQ solve from state x."""
        self.solver.run(x, Ps_warm=self._Ps, alphas_warm=self._a)
        self._Ps, self._a = self.solver._Ps, self.solver._alphas

    def nom(self, pl: int) -> float:
        """Nominal control for player `pl` at k=0."""
        return float(self.solver._best_operating_point[1][pl][0, 0])

    def B_R(self, pl: int):
        """Return (B, R0) for player `pl` at k=0."""
        xs, us = self.solver._best_operating_point
        _, Bs  = self.solver._linearize_dynamics(xs, us)          # lists len=2
        _, _, _, H = self.solver._quadraticize_costs(xs, us)      # H[pl] in (1,1,N)
        return Bs[pl][:, :, 0], H[pl][:, :, 0]

    def Zz_next(self, pl: int):
        """Value curvature and linear term at k=1."""
        Zs, zs = self.solver._best_op[4], self.solver._best_op[5]
        return Zs[pl][:, :, 1], zs[pl][:, 1]


# ╭────────────────────────────────────────────────────────────────╮
# │                     Public controller class                    │
# ╰────────────────────────────────────────────────────────────────╯
class IntersectBlameMeController:
    """
    Blame-Me/Q-MDP controller for the intersection game.

    • Maintains beliefs over opponent θ ∈ {1..5}.
    • Runs one ILQ per θ each frame (warm-started).
    • Produces a closed-form control from a belief-weighted Q-MDP mix.
    """

    def __init__(
        self,
        *,
        theta_self: int,
        ctrl_index_self: int = 0,        # 0 controls a1, 1 controls a2
        intents: Tuple[int, ...] = (1, 2, 3, 4, 5),
        # time/scales (aligned with baseline)
        dt: float = 0.1,
        horizon: int = 30,
        # cost params (aligned with your sim)
        effort_w: float = 1.0,
        b_pen: float = 1e4,
        gamma: float = 0.5,
        mu: float = 1e-6,
        v_nom: float = 18.0,
        R: float = 70.0,
        W: float = 1.5,
        L: float = 3.0,
        # constraints (aligned with baseline runner)
        acc_low: float = -10.0,
        acc_high: float = 10.0,
        # ILQ
        max_iter: int = 25,
        verbose: bool = False,
    ):
        self._intents = tuple(int(i) for i in intents)  # ensure exact order
        self._ctrl_i  = int(ctrl_index_self)
        self._theta_self = int(theta_self)

        # persistent belief over opponent θ (uniform prior)
        self._belief = np.ones(len(self._intents), dtype=np.float64)
        self._belief /= self._belief.sum()

        # store params for rebuilds
        self._params = dict(
            dt=float(dt), horizon=int(horizon),
            effort_w=float(effort_w), b_pen=float(b_pen), gamma=float(gamma),
            mu=float(mu), v_nom=float(v_nom), R=float(R), W=float(W), L=float(L),
            acc_low=float(acc_low), acc_high=float(acc_high),
            max_iter=int(max_iter), verbose=bool(verbose)
        )

        self._solvers: List[_Solver2P] = []
        self._build_solvers(theta_self=self._theta_self)

        self._log_2pi = math.log(2.0 * math.pi)

    # internal: rebuild all per-θ solvers if θ_self changes
    def _build_solvers(self, *, theta_self: int):
        p = self._params
        self._solvers = [
            _Solver2P(
                theta_self=theta_self,
                theta_opp=th,
                ctrl_index_self=self._ctrl_i,
                horizon=p["horizon"], dt=p["dt"],
                effort_w=p["effort_w"], b_pen=p["b_pen"], gamma=p["gamma"],
                mu=p["mu"], v_nom=p["v_nom"], R=p["R"], W=p["W"], L=p["L"],
                acc_low=p["acc_low"], acc_high=p["acc_high"],
                max_iter=p["max_iter"], verbose=p["verbose"]
            )
            for th in self._intents
        ]

    # ───── utils ─────
    @staticmethod
    def _clip(x, lo, hi):
        return float(np.minimum(np.maximum(x, lo), hi))

    @staticmethod
    def _safe_inv_pos(x, eps=1e-9):
        # robust inverse for (possibly tiny/negative) curvature
        return 1.0 / float(max(x, eps))

    @staticmethod
    def _curvature(B: jnp.ndarray, Z1: jnp.ndarray, R0: jnp.ndarray) -> float:
        # scalar curvature = R0 + Bᵀ Z1 B
        return float((R0 + B.T @ Z1 @ B).squeeze())

    def _log_gauss(self, u, mu, sigma2):
        # univariate Gaussian log-density
        return -0.5 * ((u - mu) ** 2) / sigma2 - 0.5 * (self._log_2pi + math.log(sigma2))

    @staticmethod
    def _softmax_logw(logw: np.ndarray) -> np.ndarray:
        # stable softmax over log-weights
        m = np.max(logw)
        if not np.isfinite(m):
            # all -inf → fallback to uniform
            return np.ones_like(logw) / float(len(logw))
        w = np.exp(logw - m)
        s = np.sum(w)
        if s <= 0.0 or not np.isfinite(s):
            return np.ones_like(logw) / float(len(logw))
        return w / s

    # ───── main API ─────
    def compute_action(
        self,
        *,
        obs: np.ndarray,
        a_opponent_observed: float,
        theta_self: Optional[int] = None
    ) -> float:
        """
        Returns our control (acceleration) using Q-MDP over opponent θ.

        Args
        ----
        obs : np.ndarray  shape (4,)  -> [d1, v1, d2, v2]
        a_opponent_observed : float   -> measured opponent acceleration
        theta_self : Optional[int]    -> if changes, rebuild solvers
        """
        if theta_self is not None and int(theta_self) != self._theta_self:
            self._theta_self = int(theta_self)
            self._build_solvers(theta_self=self._theta_self)

        x = jnp.asarray(obs, dtype=jnp.float32)
        a_opp = float(a_opponent_observed)

        # (1) Run all per-θ solvers (warm-started)
        for s in self._solvers:
            s.run(x)

        # (2) Bayes over opponent θ using opponent's action
        pl_self = self._ctrl_i
        pl_opp  = 1 - pl_self

        priors = self._belief
        logw   = np.empty_like(priors)

        for i, s in enumerate(self._solvers):
            mu_u  = s.nom(pl_opp)
            B, R0 = s.B_R(pl_opp)
            Z1, _ = s.Zz_next(pl_opp)
            curv  = self._curvature(B, Z1, R0)
            var   = self._safe_inv_pos(curv, eps=1e-9)  # σ²

            log_like = self._log_gauss(a_opp, mu_u, var)
            # NEW clipping of log-likelihoods to avoid extreme beliefs
            #log_like = jnp.clip(log_like, -10.0, 10.0)
            logw[i]  = math.log(priors[i] + 1e-30) + log_like

        self._belief = self._softmax_logw(logw)

        # (3) Q-MDP closed-form (same structure as lunar Blame-Me)
        #     u* = ( Σ pθ · ( Sθ u_nom(θ) - lθ ) ) / ( 2λ + Σ pθ · Sθ )
        eff_lambda = self._params["effort_w"]
        acc_lo     = self._params["acc_low"]
        acc_hi     = self._params["acc_high"]

        num = 0.0
        den = max(2.0 * eff_lambda, 1e-9)  # guard denominator base

        for w, s in zip(self._belief, self._solvers):
            B, R0 = s.B_R(pl_self)
            Z1, z1 = s.Zz_next(pl_self)
            S  = float((B.T @ Z1 @ B).squeeze())
            l  = float((B.T @ z1).squeeze())
            u0 = s.nom(pl_self)

            num += float(w) * (S * u0 - l)
            den += float(w) * S

        #u_cmd = self._clip(num / max(den, 1e-9), acc_lo, acc_hi)
        u_cmd = num / den
        return u_cmd

    # expose beliefs (dict and raw vector) and intent order
    @property
    def belief_over_theta(self) -> Dict[int, float]:
        return {int_th: float(p) for int_th, p in zip(self._intents, self._belief)}

    @property
    def belief_vector(self) -> np.ndarray:
        return self._belief.copy()

    @property
    def intent_order(self) -> Tuple[int, ...]:
        return self._intents
