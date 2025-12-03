#!/usr/bin/env python
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp

from intersection_blame_me_controller import _Solver2P  # required import
from iLQGame.multiplayer_dynamical_system import UncontrolledIntersection2PlayerSystem



class IntersectNoisyRationalHuman:
    """
    Noisy-rational human controller for the intersection game (effort-only form).

    Human knows its own intent θ_self and is uncertain only over robot intent θ_r ∈ intents.
    Belief over θ_r is updated via state-based Bayes using predicted next states x_{t+1}.

    Q-MDP mixture over θ_r branches (EFFORT-ONLY, NO R0):
        S_i = B_i^T Z1_i B_i
        l_i = B_i^T z1_i
        u0_i = nominal control from branch i

        μ_u  = [ Σ_i b_i * (S_i * u0_i − l_i) ] / [ 2*effort_w + Σ_i b_i * S_i ]

    Noisy-rational sampling:
        u ~ N( μ_u, 1 / (β * ( S_true + 2*effort_w )) )

    This matches your Intersection NPACE usage where effort_w=100.0 and
    denominators/precisions include +2*effort_w explicitly.
    """

    def __init__(
        self,
        *,
        theta_self: int,
        ctrl_index_self: int = 0,
        #intents: Tuple[int, ...] = (1, 2, 3, 4, 5),
        intents: Tuple[int, ...] = (1, 3, 5), # only reasonable intents for robot
        dt: float = 0.1,
        horizon: int = 30,
        effort_w: float = 1.0,
        b_pen: float = 1e4,
        gamma: float = 0.5,
        mu: float = 1e-6,
        v_nom: float = 18.0,
        R: float = 70.0,
        W: float = 1.5,
        L: float = 3.0,
        acc_low: float = -10.0,
        acc_high: float = 10.0,
        max_iter: int = 25,
        verbose: bool = False,
        beta: float = 1.0,  # inverse temperature for noisy rationality
        stochastic: bool = True,
        seed: Optional[int] = None,
        # State-based Bayes parameters (aligned with NPACE)
        beta_state: float = 1.0,
        rho_forget: float = 0.0,
        sigma2_state: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.1),
    ):
        self._intents = tuple(int(i) for i in intents)          # robot intent set
        self._ctrl_i  = int(ctrl_index_self)                    # human's control index (usually 0)
        self._theta_self = int(theta_self)                      # known human intent
        

        # initial uniform belief over robot intents
        self._belief = np.ones(len(self._intents), dtype=np.float64)
        self._belief /= self._belief.sum()

        # parameters for solvers
        self._params = dict(
            dt=float(dt), horizon=int(horizon),
            effort_w=float(effort_w), b_pen=float(b_pen), gamma=float(gamma),
            mu=float(mu), v_nom=float(v_nom), R=float(R), W=float(W), L=float(L),
            acc_low=float(acc_low), acc_high=float(acc_high),
            max_iter=int(max_iter), verbose=bool(verbose)
        )

        # inverse temperature (action noise) and state-Bayes scales
        self._beta = float(beta)
        self._beta_state = float(beta_state)
        self._stochastic = bool(stochastic)
        self._rng = np.random.default_rng(seed)

        self._rho_forget = float(rho_forget)
        self._inv_sigma2 = jnp.asarray([1.0 / s for s in sigma2_state], dtype=jnp.float32)
        self._lgv_const  = float(-0.5 * (np.sum(np.log(sigma2_state)) + 4 * math.log(2.0 * math.pi)))

        # cache predicted next states x_{t+1} per robot intent
        self._state_cache = None  # shape (nIntents, nx)

        # build ILQ solvers for each robot intent (human intent fixed)
        self._solvers: List[_Solver2P] = []
        self._build_solvers(theta_self=self._theta_self)
        self._dyn = UncontrolledIntersection2PlayerSystem(T=dt)

        # optional: external code may set self._ir_true for sampling variance branch
        self._log_2pi = math.log(2.0 * math.pi)

    def _build_solvers(self, *, theta_self: int):
        p = self._params
        self._solvers = [
            _Solver2P(
                theta_self=theta_self,             # human known intent
                theta_opp=th,                      # robot intent branch
                ctrl_index_self=self._ctrl_i,
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

    @staticmethod
    def _safe_inv_pos(x: float, eps: float = 1e-9) -> float:
        return 1.0 / float(max(x, eps))

    def set_seed(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def compute_action(
        self,
        *,
        obs: np.ndarray,
        a_opponent_observed: float,  # unused (state-based update)
        theta_self: Optional[int] = None
    ) -> float:
        # update our own intent if changed (still known perfectly)
        if theta_self is not None and int(theta_self) != self._theta_self:
            self._theta_self = int(theta_self)
            self._build_solvers(theta_self=self._theta_self)

        x = jnp.asarray(obs, dtype=jnp.float32)

        # Run ILQ for each robot-intent branch to get current linearizations
        for s in self._solvers:
            s.run(x)

        # 1) State-based Bayes over robot intents using predicted x_{t+1}
        if self._state_cache is not None:
            # err shape: (nIntents, nx)
            err  = x[None, :] - jnp.asarray(self._state_cache)
            logp = -0.5 * jnp.sum(err * err * self._inv_sigma2, axis=1)  # ∝ log N
            logp = 1.0*self._beta_state * logp + self._lgv_const  # scale + const
            logw = jnp.log(self._belief) + logp
            #new
            #logw = logw - jnp.max(logw)  # for numerical stability
            new_b = jax.nn.softmax(logw)
            # forgetting toward uniform (optional; keeps numbers aligned with NPACE)
            uniform = jnp.ones_like(new_b) / new_b.shape[0]
            #new_b = (1.0 - self._rho_forget) * new_b + self._rho_forget * uniform
            new_b = (1.0 - self._rho_forget) * new_b + self._rho_forget * uniform
            # (optional clipping for numerical hygiene)
            #new_b = jnp.clip(new_b, 1e-3, 1 - 1e-3)
            #new_b = new_b / jnp.sum(new_b)
            self._belief = np.array(new_b, dtype=np.float64)

        # 2) Q-MDP mixture for the human control (INCLUDES linear term, NO R0 in S)
        pl_self = self._ctrl_i
        nInt = len(self._solvers)

        H = np.zeros(nInt, dtype=np.float64)
        S = np.zeros(nInt, dtype=np.float64)   # S_i = B_i^T Z1_i B_i   (NO R0)
        l = np.zeros(nInt, dtype=np.float64)   # l_i = B_i^T z1_i
        u0 = np.zeros(nInt, dtype=np.float64)  # nominal u_i

        for i, s in enumerate(self._solvers):
            B_i, R0_i     = s.B_R(pl_self)          # ignore R0 (effort-only form)
            Z1_i, z1_i = s.Zz_next(pl_self)
            H[i]  = float((B_i.T @ Z1_i @ B_i).squeeze())
            S[i]  = float((R0_i + B_i.T @ Z1_i @ B_i).squeeze())
            l[i]  = float((B_i.T @ z1_i).squeeze())
            u0[i] = s.nom(pl_self)

        ###### IMPORTANT CHANGE HERE ######
        #num = float(np.dot(self._belief, H * u0 - l))
        num = float(np.dot(self._belief, S * u0 ))
        #den = 2.0 * self._params["effort_w"] + float(np.dot(self._belief, S))
        den = float(np.dot(self._belief, S))
        
        #mu_u = num / max(den, 1e-9)   # same sign convention as in NPACE
        mu_u = num / den
        #mu_u = self._clip(mu_u, self._params["acc_low"], self._params["acc_high"])
        # 3) Noisy-rational sampling: variance 1 / (β * (S_true + 2*effort_w))
        if self._stochastic:
            #ir_true = getattr(self, "_ir_true", int(np.argmax(self._belief)))
            #Bv, R0_v   = self._solvers[ir_true].B_R(pl_self)  # ignore R0
            #Z1v, _  = self._solvers[ir_true].Zz_next(pl_self)
            #S_true  = float((R0_v + Bv.T @ Z1v @ Bv).squeeze())
            #precision = self._beta * (S_true + 2.0 * self._params["effort_w"])
            S_bar    = float(np.dot(self._belief, S))               # same den as above
            #precision= self._beta * max(S_bar, 1e-19)
            precision= self._beta * S_bar
            var_u     = 1.0 / (precision)
            u_cmd     = float(mu_u + math.sqrt(abs(var_u)) * self._rng.standard_normal()) 
            #precision = self._beta * (S_true)
            #var_u     = 1.0 / max(precision, 1e-9)
            #u_cmd     = float(mu_u + math.sqrt(var_u) * self._rng.standard_normal())
        else:
            u_cmd = float(mu_u)

        #u_cmd = self._clip(u_cmd, self._params["acc_low"], self._params["acc_high"])

        # 4) Cache predicted next states for the next Bayes step
        xs_next = np.zeros((nInt, x.shape[0]), dtype=np.float32)
        for i, s in enumerate(self._solvers):
            u2_pred = s.nom(pl=1)  # predicted robot action under branch i (keep simple)
            #u2_pred = self._clip(u2_pred, self._params["acc_low"], self._params["acc_high"])
            x_next = self._dyn.disc_time_dyn(
                jnp.asarray(x, dtype=jnp.float32),
                [jnp.array([float(u_cmd)], dtype=jnp.float32),
                jnp.array([float(u2_pred)], dtype=jnp.float32)]
            )       
            xs_next[i, :] = np.asarray(x_next, dtype=np.float32)
        self._state_cache = xs_next

        return u_cmd

    @property
    def belief_over_theta(self) -> Dict[int, float]:
        return {int_th: float(p) for int_th, p in zip(self._intents, self._belief)}

    @property
    def intent_order(self) -> Tuple[int, ...]:
        return self._intents
