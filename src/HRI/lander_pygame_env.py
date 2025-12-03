#!/usr/bin/env python
# lander_pygame_env.py – 3-input version, now with a rotated thrust bar
# ================================================================
# Only two functional changes from the previous file:
#   1) Initial x-position still random (bottom-left / centre / right).
#   2) The cyan thrust bar is drawn in the **body frame**, i.e. it tilts
#      together with the lander instead of staying vertical w.r.t. screen.
#      Everything else (physics, colours, constants) is unchanged.
#
# Extra for this version:
#   • A small static “start pad” (physical, not just a sensor) is created
#     beneath the lander’s initial position each trial so the lander rests
#     on it instead of falling.  Its width is half the lander width.
# ================================================================

import pygame, pymunk, gymnasium as gym, numpy as np, random, math
from gymnasium import spaces

SPACE_DT = 1 / 30

# ───────── drawing constants ───────────────────────────────────────
SCREEN_WIDTH  = 1200
SCREEN_HEIGHT = 900
PAD_OFFSET    = 200
DASH_LENGTH   = 10
DASH_GAP      = 10

LANDER_WIDTH       = 50
LANDER_HALF_HEIGHT = LANDER_WIDTH / 2

# ───────── actuator limits ─────────────────────────────────────────
HUMAN_THRUST_MAX = 2000.0
ROBOT_DELTA_MAX  = 5000.0
TORQUE_MAX       = 1000.0

# ───────── colours ─────────────────────────────────────────────────
FLAG_COLORS      = [(255,255,0), (0,255,0), (255,0,0), (0,0,255)]
BORDER_COLOR     = (255,255,255)
STATION_COLOR    = (150,150,150)
THRUST_BAR_COLOR = (0,200,255)          # cyan

class LanderPygameEnv(gym.Env):
    metadata = {"render_mode": "human", "render_fps": 60}

    # ----------------------------------------------------------------
    def __init__(self, render_mode="human"):
        super().reset(seed=None)

        pygame.init(); pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pymunk+Pygame Lander")
        self.clock = pygame.time.Clock()
        self.font  = pygame.font.SysFont(None, 72)

        self.space         = pymunk.Space()
        self.space.gravity = (0, -800)

        self.linear_drag_coeff  = 5.0
        self.angular_drag_coeff = 100.0
        self.bounds = (0, SCREEN_WIDTH, 0, SCREEN_HEIGHT)

        self._last_thrust        = 0.0
        self._last_human_thrust  = 0.0
        self._countdown_time     = 5.0
        self.target_color        = random.choice(FLAG_COLORS)
        self.elapsed_time        = 0.0

        self.start_pad = None          # <─ track the per-trial start pad
        self._build_world()

        high = np.array([ TORQUE_MAX,
                          HUMAN_THRUST_MAX,
                          ROBOT_DELTA_MAX ], np.float32)
        self.action_space      = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (6,), np.float32)

    # ----------------------------------------------------------------
    def _build_world(self):
        # clear previous start pad if any (for successive trials)
        if self.start_pad is not None:
            self.space.remove(self.start_pad)
            self.start_pad = None

        mass, moment = 1.0, 500.0

        left   = 20 + LANDER_HALF_HEIGHT
        centre = SCREEN_WIDTH / 2
        right  = SCREEN_WIDTH - 20 - LANDER_HALF_HEIGHT
        start_x = random.choice([left, centre, right])
        start_y = LANDER_HALF_HEIGHT + 50

        self.lander = pymunk.Body(mass, moment)
        self.lander.position = (start_x, start_y)
        rect = pymunk.Poly.create_box(self.lander, (LANDER_WIDTH, LANDER_WIDTH))
        rect.friction = 0.0
        self.space.add(self.lander, rect)

        static = self.space.static_body

        # ───── new physical start pad ────────────────────────────────
        pad_half = LANDER_WIDTH / 4            # half the lander ⇒ half-width pad
        pad_y    = start_y - LANDER_HALF_HEIGHT
        self.start_pad = pymunk.Segment(static,
                                        (start_x - pad_half, pad_y),
                                        (start_x + pad_half, pad_y),
                                        5)     # radius 5 so it is drawn like stations
        self.start_pad.friction = 0.8          # some grip
        # (leave sensor = False --> real collision)
        self.space.add(self.start_pad)
        # ─────────────────────────────────────────────────────────────

        # sensor “wall” under the lander (unchanged, keeps prior logic)
        wx0 = start_x - LANDER_HALF_HEIGHT - 10
        wx1 = start_x + LANDER_HALF_HEIGHT + 10
        wy  = pad_y - 2
        wall = pymunk.Segment(static, (wx0, wy), (wx1, wy), 4)
        wall.sensor = True
        self.space.add(wall)

        # four coloured goal pads (unchanged)
        cx, cy = SCREEN_WIDTH/2, SCREEN_HEIGHT/2
        for sx, sy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
            x  = cx + sx * PAD_OFFSET
            y  = cy + sy * PAD_OFFSET
            seg = pymunk.Segment(static,
                                 (x - LANDER_WIDTH/2, y),
                                 (x + LANDER_WIDTH/2, y), 5)
            seg.sensor = True
            self.space.add(seg)

    # ----------------------------------------------------------------
    def _get_obs(self):
        x,y = self.lander.position
        vx,vy = self.lander.velocity
        ang,angv = self.lander.angle, self.lander.angular_velocity
        return np.array([x,y,vx,vy,ang,angv], np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # remove lander body & shapes
        for s in list(self.lander.shapes):
            self.space.remove(s)
        self.space.remove(self.lander)
        # remove previous start pad if still present
        if self.start_pad is not None:
            self.space.remove(self.start_pad)
            self.start_pad = None

        self._countdown_time    = 5.0
        self._last_thrust       = 0.0
        self._last_human_thrust = 0.0
        self.target_color       = random.choice(FLAG_COLORS)
        self.elapsed_time       = 0.0

        self._build_world()
        return self._get_obs(), {}

    # ----------------------------------------------------------------
    def step(self, action):
        action = np.asarray(action, np.float32).flatten()
        if action.size not in (2,3):
            raise ValueError("Action must be (τ_h, F_h) or (τ_h, F_h, ΔF_r).")
        tau_h, F_h = action[0], action[1]
        dF_r       = action[2] if action.size == 3 else 0.0

        if self._countdown_time > 0:
            self._countdown_time = max(0.0, self._countdown_time - SPACE_DT)
            return self._get_obs(), 0.0, False, False, {}

        tau_h = np.clip(tau_h, -TORQUE_MAX, TORQUE_MAX)
        F_h   = np.clip(F_h,   -HUMAN_THRUST_MAX, HUMAN_THRUST_MAX)
        dF_r  = np.clip(dF_r, -ROBOT_DELTA_MAX,   ROBOT_DELTA_MAX)
        self._last_human_thrust = float(F_h)

        F_tot = np.clip(F_h + dF_r, -10*HUMAN_THRUST_MAX, 10*HUMAN_THRUST_MAX)
        self.lander.torque = -float(tau_h)

        theta = self.lander.angle
        fx, fy = -math.sin(theta)*F_tot, math.cos(theta)*F_tot
        self.lander.apply_force_at_world_point((fx,fy), self.lander.position)
        self._last_thrust = F_tot

        vx,vy = self.lander.velocity
        self.lander.apply_force_at_world_point(
            (-self.linear_drag_coeff*vx, -self.linear_drag_coeff*vy),
            self.lander.position)
        self.lander.torque += -self.angular_drag_coeff * self.lander.angular_velocity

        self.space.step(SPACE_DT)
        self.elapsed_time += SPACE_DT

        x,y = self.lander.position
        l,r,b,t = self.bounds
        done = not (l <= x <= r and b <= y <= t)
        return self._get_obs(), 0.0, done, False, {}

    # ----------------------------------------------------------------
    def render(self):
        self.screen.fill((10,10,40))
        pygame.draw.rect(self.screen, BORDER_COLOR,
                         (0,0,SCREEN_WIDTH,SCREEN_HEIGHT), 2)
        pygame.draw.rect(self.screen, self.target_color,
                         pygame.Rect(10,10,40,40))

        cx,cy = SCREEN_WIDTH/2, SCREEN_HEIGHT/2
        left_x,right_x = cx-PAD_OFFSET, cx+PAD_OFFSET
        bottom_y = SCREEN_HEIGHT-(cy-PAD_OFFSET)
        top_y    = SCREEN_HEIGHT-(cy+PAD_OFFSET)

        for x_dash,col in [(left_x,FLAG_COLORS[0]), (right_x,FLAG_COLORS[1])]:
            y0=0
            while y0<SCREEN_HEIGHT:
                pygame.draw.line(self.screen,col,(x_dash,y0),
                                 (x_dash,min(y0+DASH_LENGTH,SCREEN_HEIGHT)),1)
                y0 += DASH_LENGTH + DASH_GAP
        for y_dash,col in [(bottom_y,FLAG_COLORS[2]), (top_y,FLAG_COLORS[3])]:
            x0=0
            while x0<SCREEN_WIDTH:
                pygame.draw.line(self.screen,col,(x0,y_dash),
                                 (min(x0+DASH_LENGTH,SCREEN_WIDTH),y_dash),1)
                x0 += DASH_LENGTH + DASH_GAP

        if self._countdown_time > 0:
            n = math.ceil(self._countdown_time)
            txt = self.font.render(str(n), True, (255,255,255))
            w,h = txt.get_size()
            self.screen.blit(txt, ((SCREEN_WIDTH-w)//2,(SCREEN_HEIGHT-h)//2))
            pygame.display.flip(); self.clock.tick(60); return

        for seg in [s for s in self.space.static_body.shapes
                    if isinstance(s,pymunk.Segment) and s.radius==5]:
            (x0,y0),(x1,_) = seg.a, seg.b
            screen_y = SCREEN_HEIGHT - y0
            pygame.draw.line(self.screen, STATION_COLOR,
                             (x0,screen_y),(x1,screen_y),5)
            cx_line = (x0+x1)/2
            cy_line = y0 + LANDER_HALF_HEIGHT
            pygame.draw.circle(self.screen, STATION_COLOR,
                               (int(cx_line), int(SCREEN_HEIGHT-cy_line)),8)

        poly = next((s for s in self.lander.shapes if isinstance(s,pymunk.Poly)), None)
        if poly:
            pts=[self.lander.local_to_world(v) for v in poly.get_vertices()]
            pts=[(int(p.x), int(SCREEN_HEIGHT-p.y)) for p in pts]
            pygame.draw.polygon(self.screen, (255,165,0), pts)

            # ---- rotated thrust bar ----
            max_abs = 1000.0
            ratio   = max(-1.0, min(1.0, self._last_human_thrust / max_abs))
            if abs(ratio) > 1e-3:
                bar_half_len = LANDER_HALF_HEIGHT - 4
                fill_len     = abs(ratio) * bar_half_len
                sign         = 1.0 if ratio > 0 else -1.0

                # local +y axis in world coords
                theta = self.lander.angle
                dir_x, dir_y = -math.sin(theta),  math.cos(theta)
                perp_x, perp_y =  dir_y, -dir_x   # 90° CCW

                # start point at body centre
                cx_w, cy_w = self.lander.position
                ex_w = cx_w + dir_x * sign * fill_len
                ey_w = cy_w + dir_y * sign * fill_len

                half_w = 4                       # bar half-width in px
                p1 = (cx_w + perp_x*half_w, cy_w + perp_y*half_w)
                p2 = (cx_w - perp_x*half_w, cy_w - perp_y*half_w)
                p3 = (ex_w - perp_x*half_w, ey_w - perp_y*half_w)
                p4 = (ex_w + perp_x*half_w, ey_w + perp_y*half_w)

                pts_screen = [(int(p[0]), int(SCREEN_HEIGHT-p[1]))
                              for p in (p1,p2,p3,p4)]
                pygame.draw.polygon(self.screen, THRUST_BAR_COLOR, pts_screen)

        if self._last_thrust > 0:
            nozzle = self.lander.local_to_world((0,-LANDER_HALF_HEIGHT))
            pygame.draw.circle(self.screen,(255,0,0),
                               (int(nozzle.x),int(SCREEN_HEIGHT-nozzle.y)),10)

        pygame.display.flip(); self.clock.tick(30)

    # ----------------------------------------------------------------
    def close(self):
        pygame.quit()
