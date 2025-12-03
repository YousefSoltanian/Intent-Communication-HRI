#!/usr/bin/env python
# goal_lander_env.py
# ---------------------------------------------------------------------
import math, numpy as np, pygame
from lander_pygame_env import (
    LanderPygameEnv,
    FLAG_COLORS,
    LANDER_HALF_HEIGHT,
    PAD_OFFSET,
)

# ---------------- tolerances ----------------------------------------
COM_TOL, GROUND_TOL = 20, 15          # px
UPRIGHT             = math.radians(15)
VEL_TOL             = 30              # px/s

class GoalLanderEnv(LanderPygameEnv):
    """
    Four-corner landing-pad environment.

       • Safe landing on designated pad → success
       • Safe landing on any other pad  → failure
       • Going out of bounds            → failure
    """

    def __init__(self, *, show_x_goal: bool = True, show_y_goal: bool = False):
        super().__init__(render_mode="human")
        self.show_x_goal = show_x_goal
        self.show_y_goal = show_y_goal
        self.goal_idx    = (0, 0)       # (x_idx, y_idx)
        self._bg_color   = (10, 10, 40)
        self.outcome     = None         # "success" | "failure"

    # --------------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)

        rng = self.np_random                # gymnasium's RNG
        self.goal_idx  = (int(rng.integers(0, 2)),
                          int(rng.integers(0, 2)))
        self.goal_x_px = self._pad_x(self.goal_idx[0])
        self.goal_y_px = self._pad_y(self.goal_idx[1])

        self.outcome   = None
        info["goal_idx"] = self.goal_idx    # expose for debug / logging
        return obs, info

    # --------------------------------------------------------------- #
    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)

        # out-of-bounds ⇒ failure
        if done and self.outcome is None:
            self.outcome = "failure"

        # evaluate pads after countdown
        if not done and self._countdown_time <= 0:
            if self._landed_on_pad(obs, *self.goal_idx):
                done, self.outcome = True, "success"
            elif self._failure(obs):
                done, self.outcome = True, "failure"

        if done:
            info["outcome"]  = self.outcome
            info["goal_idx"] = self.goal_idx
        return obs, reward, done, trunc, info

    # ---------------- helpers --------------------------------------
    def _failure(self, obs):
        for xi in (0, 1):
            for yi in (0, 1):
                if (xi, yi) == self.goal_idx:
                    continue
                if self._landed_on_pad(obs, xi, yi):
                    return True
        return False

    def _landed_on_pad(self, obs, pad_x_idx, pad_y_idx):
        x, y, _, vy, angle, _ = obs
        pad_x = self._pad_x(pad_x_idx)
        pad_y = self._pad_y(pad_y_idx)

        if abs(x - pad_x) > COM_TOL:                           return False
        if abs(y - (pad_y + LANDER_HALF_HEIGHT)) > COM_TOL:    return False
        if abs((y - LANDER_HALF_HEIGHT) - pad_y) > GROUND_TOL: return False
        if abs(angle) > UPRIGHT or abs(vy) > VEL_TOL:          return False
        return True

    # ---------------- coordinate helpers ---------------------------
    def _pad_x(self, x_idx):  # 0 = left, 1 = right
        cx = self.bounds[1] / 2
        return cx - PAD_OFFSET if x_idx == 0 else cx + PAD_OFFSET

    def _pad_y(self, y_idx):  # 0 = low, 1 = high
        cy = self.bounds[3] / 2
        return cy - PAD_OFFSET if y_idx == 0 else cy + PAD_OFFSET

    # ---------------- render overlay -------------------------------
    def render(self):
        saved = self.target_color
        self.target_color = self._bg_color
        super().render()
        self.target_color = saved

        base_x, offset = 60, 0
        if self.show_x_goal:
            pygame.draw.rect(self.screen, FLAG_COLORS[self.goal_idx[0]],
                             pygame.Rect(base_x + offset, 10, 16, 16))
            offset += 20
        if self.show_y_goal:
            pygame.draw.rect(self.screen, FLAG_COLORS[2 + self.goal_idx[1]],
                             pygame.Rect(base_x + offset, 10, 16, 16))
            offset += 20
        if offset:
            pygame.display.update(pygame.Rect(base_x, 10, offset, 16))
