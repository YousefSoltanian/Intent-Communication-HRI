# demo_ct.py
import sys, pygame, numpy as np, math
from pathlib import Path
from lander_pygame_env import LanderPygameEnv

T_MAX   = 1000.0    # matches env high
TORQUE_MAX = 1000.0 # matches env high
THROTTLE_STEP = 0.03
OMEGA_STEP = 1            # rad/s per frame
KP = 150.0                   # proportional gain → torque (tune)
KD = 2.0                    # damping gain → torque

env = LanderPygameEnv(render_mode="human")
obs, _ = env.reset()

throttle = 0.0               # [0,1]
omega_target = 0.0           # desired angular velocity

pygame.init()
clock = pygame.time.Clock()
running = True
while running:
    dt = clock.tick(60) * 0.001
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    keys = pygame.key.get_pressed()

    # --- increment throttle (Up / Down) ---
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        throttle = min(1.0, throttle + THROTTLE_STEP)
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
        throttle = max(0.0, throttle - THROTTLE_STEP)

    # --- increment desired angular velocity (Left / Right) ---
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        omega_target +=  OMEGA_STEP
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        omega_target -=  OMEGA_STEP
    else:
        # gentle auto‑decay toward zero for stability
        omega_target *= 0.97

    # clamp target ω so PD doesn’t saturate
    omega_target = np.clip(omega_target, -3.0, 3.0)

    # Current angular velocity from last obs
    omega = obs[5]
    # PD torque toward omega_target, then clip to env limits
    torque = -np.clip(KP*(omega_target - omega) - KD*omega,
                     -TORQUE_MAX, TORQUE_MAX)

    thrust = throttle * T_MAX

    obs, _, done, _, _ = env.step([torque, thrust])
    env.render()
    if done:
        print("Out of bounds or landed – Esc to quit.")
pygame.quit()
