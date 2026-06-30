# test_cnn.py (AI Test with Advanced Status & Speed Dashboard)
import cv2
import numpy as np
import gymnasium as gym
import pygame
from stable_baselines3 import PPO
from gymnasium.wrappers import FrameStackObservation

# Import our unified wrappers
from lunar_lander_wrappers import StateCaptureWrapper, CustomRewardWrapper, VisionObservationWrapper

if __name__ == "__main__":
    raw_env = gym.make("LunarLander-v3", render_mode="rgb_array")

    # Onion Architecture for Testing
    env_state = StateCaptureWrapper(raw_env)
    env_reward = CustomRewardWrapper(env_state)
    env_vision = VisionObservationWrapper(env_reward)
    env_stacked = FrameStackObservation(env_vision, stack_size=4)

    model_path = "models_cnn/ppo_cnn_20260701_024739/ppo_cnn_20260701_024739_final_3000000_steps"
    model = PPO.load(model_path)

    pygame.init()
    screen_width, screen_height = 600, 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Lunar Lander v3")
    clock = pygame.time.Clock()

    score_font = pygame.font.SysFont(None, 32)
    # efficiency_font = pygame.font.SysFont(None, 24)
    telemetry_font = pygame.font.SysFont("Consolas", 20)
    result_font = pygame.font.SysFont(None, 64)
    small_font = pygame.font.SysFont(None, 24)

    obs, info = env_stacked.reset(seed=42)
    running = True
    game_over = False

    message = ""
    msg_color = (255, 255, 255)
    shaped_reward_sum = 0.0
    raw_reward_sum = 0.0
    step_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game_over:
            # AI Takes Control Here!
            action, _states = model.predict(obs, deterministic=True)
            obs, shaped_reward, terminated, truncated, info = env_stacked.step(action.item())
            step_count += 1

            shaped_reward_sum += shaped_reward
            raw_reward_sum += env_state.raw_reward

            # --- Advanced End-State Classification ---
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
        # Render Graphics
        # ==========================================
        rgb_array = raw_env.render()
        surf = pygame.surfarray.make_surface(np.swapaxes(rgb_array, 0, 1))
        screen.blit(surf, (0, 0))

        if game_over:
            overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))

        # --- Draw Dual Scores ---
        # steps_surface = score_font.render(f"Steps: {step_count}", True, (150, 200, 255))
        # raw_surface = score_font.render(f"Game Score:   {raw_reward_sum:.1f}", True, (255, 200, 50))
        # shaped_surface = score_font.render(f"Shaped Score: {shaped_reward_sum:.1f}", True, (100, 255, 255))
        # screen.blit(steps_surface, (10, 10))
        # screen.blit(raw_surface, (10, 40))
        # screen.blit(shaped_surface, (10, 70))

        # steps_surface = score_font.render(f"Steps: {step_count}", True, (150, 200, 255))
        # raw_surface = score_font.render(f"Game Score:   {raw_reward_sum:.1f}", True, (255, 200, 50))
        shaped_surface = score_font.render(f"Score: {shaped_reward_sum:.1f}", True, (100, 255, 255))
        # screen.blit(steps_surface, (10, 10))
        # screen.blit(raw_surface, (10, 40))
        screen.blit(shaped_surface, (10, 10))

        # --- Draw Efficiency Metrics ---
        # main_fuel_surface = efficiency_font.render(f"Main Engine Bill: -{env_state.main_fuel_bill:.1f}", True, (255, 150, 150))
        # side_fuel_surface = efficiency_font.render(f"Side Engine Bill: -{env_state.side_fuel_bill:.1f}", True, (255, 200, 150))

        # screen.blit(steps_surface, (10, 75))
        # screen.blit(main_fuel_surface, (10, 100))
        # screen.blit(side_fuel_surface, (10, 125))

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
            f"R-Leg:  {'TOUCH' if raw_telemetry[7] == 1.0 else 'AIR'}"
        ]

        for idx, text in enumerate(telemetry_labels):
            text_color = (200, 200, 200)

            if "TOUCH" in text:
                text_color = (100, 255, 100)
            elif idx in [2, 3]:
                text_color = (255, 150, 50) if abs(raw_telemetry[idx]) > 0.2 else (100, 255, 100)

            telemetry_surface = telemetry_font.render(text, True, text_color)
            screen.blit(telemetry_surface, (450, 10 + idx * 22))

        # --- Draw Vertical Speed Monitor ---
        current_y_vel = raw_telemetry[3]
        vel_color = (100, 255, 100) if current_y_vel >= -0.2 else (255, 100, 100)
        speed_surface = telemetry_font.render(f"V-Spd:  {current_y_vel:.2f}", True, vel_color)
        screen.blit(speed_surface, (450, 10 + 8 * 22))

        # --- Game Over Logic ---
        if game_over:
            result_surface = result_font.render(message, True, msg_color)
            text_rect = result_surface.get_rect(center=(screen_width // 2, screen_height // 2))
            screen.blit(result_surface, text_rect)

            pygame.display.flip()

            # AI doesn't need to press space; automatically restart after 2 seconds
            pygame.time.wait(2000)
            obs, info = env_stacked.reset()
            game_over = False
            message = ""
            msg_color = (255, 255, 255)
            shaped_reward_sum = 0.0
            raw_reward_sum = 0.0
            step_count = 0
            continue

        pygame.display.flip()
        clock.tick(50)

    env_stacked.close()
    pygame.quit()
