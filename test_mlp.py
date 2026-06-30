# test_mlp.py
import cv2
import numpy as np
import pygame
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO

# Import our unified wrappers
from lunar_lander_wrappers import MlpStateCaptureWrapper, MlpCustomRewardWrapper

if __name__ == "__main__":
    # ==========================================
    # 1. Initialize Environment with Wrappers
    # ==========================================
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env_state = MlpStateCaptureWrapper(env)
    env = MlpCustomRewardWrapper(env_state)
    obs, info = env.reset()

    model_path = "models_mlp/ppo_mlp_20260702_212701_seed_21062/ppo_mlp_20260702_212701_seed_21062_final_20000000_steps.zip"
    model = PPO.load(model_path)

    # ==========================================
    # 2. Initialize Pygame & Fonts
    # ==========================================
    pygame.init()
    screen_width, screen_height = 600, 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Lunar Lander v3")
    clock = pygame.time.Clock()
    score_font = pygame.font.SysFont(None, 32)
    telemetry_font = pygame.font.SysFont("Consolas", 20)
    result_font = pygame.font.SysFont(None, 64)
    small_font = pygame.font.SysFont(None, 24)

    # ==========================================
    # 3. Game Loop Variables
    # ==========================================

    running = True
    game_over = False
    message = ""
    msg_color = (255, 255, 255)

    raw_reward_sum = 0.0
    shaped_reward_sum = 0.0
    step_count = 0
    action_map = {0: "NONE", 1: "LEFT", 2: "UP", 3: "RIGHT"}
    current_action_text = "NONE"

    # ==========================================
    # 4. Main Game Loop
    # ==========================================
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game_over:
            # AI Takes Control Here!
            action, _states = model.predict(obs, deterministic=True)
            current_action_text = action_map.get(action.item())
            obs, reward, terminated, truncated, info = env.step(action.item())
            step_count += 1
            shaped_reward_sum += reward

            # --- End-State Classification ---
            raw_reward_sum = env_state.raw_reward_sum
            if terminated or truncated:
                game_over = True
                current_x_pos = env_state.last_raw_obs[0]

                if truncated and not terminated:
                    message = "MISSION TIMEOUT!"
                    msg_color = (255, 200, 50)
                elif abs(current_x_pos) >= 1.0:
                    message = "OUT OF BOUNDS!"
                    msg_color = (255, 50, 50)
                elif raw_reward_sum >= 200.0:
                    message = "PERFECT LANDING"
                    msg_color = (50, 255, 50)
                elif raw_reward_sum > 0.0:
                    message = "SAFE LANDING"
                    msg_color = (150, 200, 255)
                else:
                    message = "CRASHED!"
                    msg_color = (255, 50, 50)

        # ==========================================
        # 5. Render Graphics
        # ==========================================
        rgb_array = env.render()
        surf = pygame.surfarray.make_surface(np.swapaxes(rgb_array, 0, 1))
        screen.blit(surf, (0, 0))

        if game_over:
            overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))

        # --- Draw Scores ---
        raw_surface = score_font.render(f"Raw Score: {raw_reward_sum:.1f}", True, (255, 200, 50))
        screen.blit(raw_surface, (10, 10))
        shaped_surface = score_font.render(f"Shaped Score: {shaped_reward_sum:.1f}", True, (100, 255, 255))
        screen.blit(shaped_surface, (10, 40))

        # --- Draw Telemetry ---
        raw_telemetry = env_state.last_raw_obs
        telemetry_labels = [
            f"X-Pos:  {raw_telemetry[0]:.2f}",
            f"Y-Pos:  {raw_telemetry[1]:.2f}",
            f"X-Vel:  {raw_telemetry[2]:.2f}",
            f"Y-Vel:  {raw_telemetry[3]:.2f}",
            f"Angle:  {raw_telemetry[4]:.2f}",
            f"AngVel: {raw_telemetry[5]:.2f}",
            f"L-Leg:  {'TOUCH' if raw_telemetry[6] == 1.0 else 'AIR'}",
            f"R-Leg:  {'TOUCH' if raw_telemetry[7] == 1.0 else 'AIR'}",
            f"ACT:    {current_action_text}",
        ]

        for idx, text in enumerate(telemetry_labels):
            text_color = (200, 200, 200)

            if "TOUCH" in text:
                text_color = (100, 255, 100)
            elif idx in [2, 3]:
                text_color = (255, 150, 50) if abs(raw_telemetry[idx]) > 0.2 else (100, 255, 100)
            elif "ACT" in text:
                text_color = (255, 100, 100) if "NONE" not in text else (200, 200, 200)

            telemetry_surface = telemetry_font.render(text, True, text_color)
            screen.blit(telemetry_surface, (450, 10 + idx * 22))

        # --- Game Over ---
        if game_over:
            result_surface = result_font.render(message, True, msg_color)
            text_rect = result_surface.get_rect(center=(screen_width // 2, screen_height // 2))
            screen.blit(result_surface, text_rect)

            pygame.display.flip()

            # AI doesn't need to press space; automatically restart after 2 seconds
            pygame.time.wait(2000)
            obs, info = env.reset()
            game_over = False
            message = ""
            msg_color = (255, 255, 255)
            raw_reward_sum = 0.0
            shaped_reward_sum = 0.0
            step_count = 0
            continue

        pygame.display.flip()
        clock.tick(50)

    # ==========================================
    # 6. Cleanup
    # ==========================================
    env.close()
    pygame.quit()
