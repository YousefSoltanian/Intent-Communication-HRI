#!/usr/bin/env python
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp

# Same solver you used for the intersection human:
from intersection_blame_me_controller import _Solver2P  # required import
from iLQGame.multiplayer_dynamical_system import UncontrolledIntersection2PlayerSystem


class IntersectBlameMeController:
    """
    Blame-Me / Q-MDP robot for the intersection game (scalar control: acceleration).

    Robot knows its own intent θ_self and is uncertain only over HUMAN intent θ_h ∈ intents.
    Belief over θ_h is updated via *action*-likelihood Bayes using the observed human action u_h.

    Prediction model for human action under each θ_h branch (1-D):
        For branch i (θ_h = intents[i]), with current linearization:
            H1_i = B1^T Z1 B1         (scalar)
            S1_i = R01 + H1_i         (scalar)
            l1_i = B1^T z1            (scalar)
            u0_i = nominal human action from branch i (scalar)

        Predicted mean human action:
            μ1_i = (H1_i * u0_i − l1_i) / (S1_i + 2*effort_w_qmdp)

        Policy precision for Bayes:
            Prec_policy_i = beta_action_like * (S1_i + 2*effort_w_qmdp)

        Observation model (to avoid over-sharp posteriors):
            Σ_total_i = (Prec_policy_i)^(-1) + σ_obs^2
            Prec_total_i = 1 / Σ_total_i

        log p(u_h | θ_h=i) = 0.5 log Prec_total_i − 0.5 Prec_total_i * (u_h − μ1_i)^2

    Robot Q-MDP control (with cross coupling to human deviation):
        For each θ_h branch:
            H2 = B2^T Z2 B2
            S2 = R02 + H2
            ℓ2 = B2^T z2
            C  = (B2^T Z2 B1 + B1^T Z2 B2)/2        (symmetric scalar)

            δu1 = μ1_i − u1_nom_true                 (human deviation vs nominal)

            bias_i = H2 * u2_nom_true − (ℓ2 + C * δu1)

        Mix over robot belief b(θ_h):
            S̄ = Σ_i b_i S2_i
            num = Σ_i b_i * bias_i

            (S̄ + 2*effort_w_qmdp) * u2 = num    →  u2 = num / (S̄ + 2*effort_w_qmdp)
    """

    def __init__(
        self,
        *,
        theta_self: int,                 # robot's true intent (index into robot intent set used by _Solver2P)
        ctrl_index_self: int = 1,        # robot controls player-1
        intents: Tuple[int, ...] = (1, 5, 10),  # HUMAN intent set (what robot is inferring)
        dt: float = 0.1,
        horizon: int = 30,
        effort_w: float = 100.0,         # passes into solver costs (effort in player's R0)
        b_pen: float = 1e4,
        gamma: float = 0.5,
        mu: float = 1e-6,
        v_nom: float = 18.0,
        R: float = 70.0,
        W: float = 1.5,
        L: float = 3.0,
        acc_low: float = -50.0,
        acc_high: float = 50.0,
        max_iter: int = 25,
        verbose: bool = False,

        # Bayes on action likelihood (softness + observation noise)
        beta_action_like: float = 1.0,   # scales policy precision used in the likelihood
        sigma2_action_obs: float = 1e-5, # σ_obs^2 added to policy covariance (keeps it soft)

        # Optional extra effort ridge in Q-MDP denominators (stability knob)
        effort_w_qmdp: float = 0.0,

        # Robot can be deterministic; keep stochastic path for completeness
        beta: float = 1.0,
        stochastic: bool = False,
        seed: Optional[int] = None,
    ):
        # intents over HUMAN behavior (what the robot is inferring)
        self._intents = tuple(int(i) for i in intents)
        self._ctrl_i  = int(ctrl_index_self)
        self._theta_self = int(theta_self)

        # uniform prior over human intents
        self._belief = np.ones(len(self._intents), dtype=np.float64)
        self._belief /= self._belief.sum()

        self._params = dict(
            dt=float(dt), horizon=int(horizon),
            effort_w=float(effort_w), b_pen=float(b_pen), gamma=float(gamma),
            mu=float(mu), v_nom=float(v_nom), R=float(R), W=float(W), L=float(L),
            acc_low=float(acc_low), acc_high=float(acc_high),
            max_iter=int(max_iter), verbose=bool(verbose)
        )

        # likelihood + Q-MDP knobs
        self._beta_like = float(beta_action_like)
        self._sigma2_obs = float(sigma2_action_obs)
        self._effort_w_qmdp = float(effort_w_qmdp)

        self._beta = float(beta)
        self._stochastic = bool(stochastic)
        self._rng = np.random.default_rng(seed)

        # Build one solver per HUMAN intent; robot intent fixed = theta_self
        self._solvers: List[_Solver2P] = []
        self._build_solvers(theta_self=self._theta_self)
        self._dyn = UncontrolledIntersection2PlayerSystem(T=dt)

        # Cache for predicted (μ1_i, Prec_total_i) used by Bayes at the *next* step
        self._pred_ctrl = None  # dict with "mu_u1": (nH,), "Prec_u1": (nH,)

    @staticmethod
    def _scalar(x) -> float:
        """Convert (possibly JAX) arrays of shape (), (1,), or (1,1) to Python float."""
        return float(np.asarray(x).squeeze())

    def _build_solvers(self, *, theta_self: int):
        p = self._params
        # θ_self is the robot intent for the player's own cost (player 1).
        # θ_opp enumerates HUMAN intents for the opponent.
        self._solvers = [
            _Solver2P(
                theta_self=theta_self,            # robot known intent
                theta_opp=int(th),                # HUMAN intent branch
                ctrl_index_self=self._ctrl_i,     # robot controls player-1
                horizon=p["horizon"], dt=p["dt"],
                effort_w=p["effort_w"], b_pen=p["b_pen"], gamma=p["gamma"],
                mu=p["mu"], v_nom=p["v_nom"], R=p["R"], W=p["W"], L=p["L"],
                acc_low=p["acc_low"], acc_high=p["acc_high"],
                max_iter=p["max_iter"], verbose=p["verbose"]
            )
            for th in self._intents
        ]

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return float(np.minimum(np.maximum(x, lo), hi))

    # Exposed for logging/inspection (robot's belief over HUMAN intents)
    @property
    def belief_over_theta(self) -> Dict[int, float]:
        return {int_th: float(p) for int_th, p in zip(self._intents, self._belief)}

    def compute_action(
        self,
        *,
        obs: np.ndarray,
        a_opponent_observed: float,   # observed human acceleration (scalar)
        theta_self: Optional[int] = None
    ) -> float:
        # Update robot's own intent if changed
        if theta_self is not None and int(theta_self) != self._theta_self:
            self._theta_self = int(theta_self)
            self._build_solvers(theta_self=self._theta_self)

        x = jnp.asarray(obs, dtype=jnp.float32)

        # 0) Run each HUMAN-intent branch solver
        for s in self._solvers:
            s.run(x)

        nH = len(self._solvers)

        # ---- (A) Bayes over HUMAN intents using observed human action u_h ----
        if self._pred_ctrl is not None:
            mu_arr   = np.asarray(self._pred_ctrl["mu_u1"],   dtype=float).reshape(nH,)
            Prec_arr = np.asarray(self._pred_ctrl["Prec_u1"], dtype=float).reshape(nH,)
            u_obs = float(a_opponent_observed)

            # log N(u | μ, Σ_total)  with Σ_total implicitly baked into Prec_arr
            log_like = 0.5*np.log(np.maximum(Prec_arr, 1e-18)) - 0.5*Prec_arr*(u_obs - mu_arr)**2

            log_post = np.log(np.asarray(self._belief) + 1e-18) + log_like
            log_post = log_post - np.max(log_post)
            post = jax.nn.softmax(jnp.asarray(log_post)).astype(jnp.float64)
            self._belief = np.array(post, dtype=np.float64)

        # ---- (B) Prepare human predicted policy per HUMAN-intent branch ----
        mu1 = np.zeros(nH, dtype=np.float64)           # predicted human mean action
        Prec_total = np.zeros(nH, dtype=np.float64)    # total precision for likelihood

        # Robot Q-MDP terms
        S2   = np.zeros(nH, dtype=np.float64)          # denominators per branch
        bias = np.zeros(nH, dtype=np.float64)          # numerators per branch

        for ih, s in enumerate(self._solvers):
            # Human (player 0)
            B1, R01 = s.B_R(pl=0)
            Z1, z1  = s.Zz_next(pl=0)
            H1      = self._scalar(B1.T @ Z1 @ B1)
            R01_s   = self._scalar(R01)
            S1      = R01_s + H1
            l1      = self._scalar(B1.T @ z1)
            u1_nom  = self._scalar(s.nom(pl=0))

            denom_h = S1 + 2.0*self._effort_w_qmdp
            denom_h = max(denom_h, 1e-18)
            mu1_raw = (H1 * u1_nom - l1) / denom_h
            #mu1[ih] = self._clip(mu1_raw, self._params["acc_low"], self._params["acc_high"])
            mu1[ih] = mu1_raw

            Prec_policy   = self._beta_like * denom_h
            Sigma_total   = (1.0 / Prec_policy) + self._sigma2_obs
            Prec_total[ih]= 1.0 / max(Sigma_total, 1e-18)

            # Robot (player 1)
            B2, R02 = s.B_R(pl=1)
            Z2, z2  = s.Zz_next(pl=1)
            H2      = self._scalar(B2.T @ Z2 @ B2)
            R02_s   = self._scalar(R02)
            S2[ih]  = R02_s + H2
            l2      = self._scalar(B2.T @ z2)
            u2_nom  = self._scalar(s.nom(pl=1))

            # Cross-term C (symmetric)
            C = 0.5 * ( self._scalar(B2.T @ Z2 @ B1) + self._scalar(B1.T @ Z2 @ B2) )

            delta_u1 = mu1[ih] - u1_nom
            l2_adj   = l2 + C * float(delta_u1)

            bias[ih] = H2 * u2_nom - l2_adj

        # ---- (C) Robot Q-MDP mixing over HUMAN-intent belief ----
        b     = np.asarray(self._belief, dtype=np.float64).reshape(nH,)
        S_bar = float(np.dot(b, S2))
        num   = float(np.dot(b, bias))

        S_bar += 2.0*self._effort_w_qmdp
        u2 = num / max(S_bar, 1e-18)

        if self._stochastic:
            prec = self._beta * max(S_bar, 1e-18)
            var  = 1.0 / prec
            u2   = float(u2 + math.sqrt(var) * self._rng.standard_normal())

        #u2 = self._clip(float(u2), self._params["acc_low"], self._params["acc_high"])

        # ---- (D) Cache human-policy stats for next-step Bayes ----
        self._pred_ctrl = {"mu_u1": mu1.copy(), "Prec_u1": Prec_total.copy()}

        return float(u2)
