#!/usr/bin/env python
from __future__ import annotations
import os, sys, numpy as np, jax.numpy as jnp
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from iLQGame.cost        import ThrustPlayerParamCost, TorquePlayerParamCost
from iLQGame.player_cost import PlayerCost
from iLQGame.constraint  import BoxConstraint
from iLQGame.ilq_solver  import ILQSolver
from iLQGame.multiplayer_dynamical_system import LunarLander3PlayerSystem

# ---------- shared weights ------------------------------------------
T_EFF, TAU_EFF = 0.1, 9.0
W_TRUE = jnp.array([968., 962., 1036., 1035.,  T_EFF])
Q_TRUE = jnp.array([971., 972., 1037.,  967., TAU_EFF])

# ---------- constants -----------------------------------------------
DT          = 1/60
HORIZON     = 60
DELTA_LIMIT = 300.0                 # matches env's ROBOT_DELTA_MAX
HOVER_BIAS  = 900.0                 # supplied by runner via F_h
LANDER_HALF_HEIGHT = 25.0

ROBOT_BOX  = BoxConstraint(-DELTA_LIMIT, DELTA_LIMIT)
HUMAN_BOX  = BoxConstraint(0.0, 0.0)   # F_h fixed (runner sets it)
TORQUE_BOX = BoxConstraint(0.0, 0.0)   # τ fixed

class ILQGameThrustController:
    """Outputs only Δ-thrust (player-1)."""

    def __init__(self, goal_xy, dt=DT, horizon=HORIZON, max_iter=15):
        self.dynamics = LunarLander3PlayerSystem(T=dt)
        self.solver   = self._build_solver(goal_xy, horizon, max_iter)

    # -----------------------------------------------------------------
    def _build_solver(self, goal_xy, H, max_iter):
        gx, gy = goal_xy; target = (gx, gy + LANDER_HALF_HEIGHT)

        pc_h   = PlayerCost(); pc_r = PlayerCost(); pc_tau = PlayerCost()
        pc_h  .add_cost(ThrustPlayerParamCost(W_TRUE, target, horizon=H), arg="x")
        pc_r  .add_cost(ThrustPlayerParamCost(W_TRUE, target, horizon=H), arg="x")
        pc_tau.add_cost(TorquePlayerParamCost(Q_TRUE, target, horizon=H), arg="x")

        zeros_P = jnp.zeros((1, 6, H))
        zeros_a = jnp.zeros((1,   H))

        return ILQSolver(
            dynamics      = self.dynamics,
            player_costs  = [pc_h, pc_r, pc_tau],
            Ps            = [zeros_P, zeros_P, zeros_P],
            alphas        = [zeros_a, zeros_a, zeros_a],   # no hover bias inside game
            max_iter      = max_iter,
            u_constraints = [HUMAN_BOX, ROBOT_BOX, TORQUE_BOX],
            verbose       = False,
        )

    # -----------------------------------------------------------------
    def compute_action(self, obs: np.ndarray) -> float:
        self.solver.run(jnp.asarray(obs, dtype=jnp.float32))
        delta = float(self.solver._best_operating_point[1][1][0, 0])
        return np.clip(delta, -DELTA_LIMIT, DELTA_LIMIT)
