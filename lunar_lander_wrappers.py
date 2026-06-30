# lunar_lander_wrappers.py
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MlpStateCaptureWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_raw_obs = np.zeros(8)
        self.saved_raw_reward_sum = 0.0
        self.raw_reward_sum = 0.0
        # self.raw_reward = 0.0

        self.saved_ep_step_count = 0
        self.ep_step_count = 0

    def reset(self, **kwargs):
        self.saved_raw_reward_sum = self.raw_reward_sum
        self.raw_reward_sum = 0.0

        self.saved_ep_step_count = self.ep_step_count
        self.ep_step_count = 0

        obs, info = self.env.reset(**kwargs)
        self.last_raw_obs = obs
        # self.raw_reward = 0.0

        return obs, info

    def step(self, action):
        prev_raw_obs = self.last_raw_obs.copy()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.raw_reward_sum += reward
        self.ep_step_count += 1

        if terminated or truncated:
            self.last_ep_raw_reward = self.raw_reward_sum
            self.saved_ep_step_count = self.ep_step_count

        # --- Anti-Spike Filter for Dashboard UI ---
        if terminated:
            temp_ops = obs.copy()
            temp_ops[2] = prev_raw_obs[2]   # Freeze X-Vel
            temp_ops[3] = prev_raw_obs[3]   # Freeze Y-Vel
            temp_ops[5] = prev_raw_obs[5]   # Freeze AngVel
            self.last_raw_obs = temp_ops
        else:
            self.last_raw_obs = obs

        return obs, reward, terminated, truncated, info


class MlpCustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_v_y = 0.0
        self.prev_v_x = 0.0
        self.prev_distance = 0.0

        self.saved_shaped_reward_sum = 0.0
        self.shaped_reward_sum = 0.0

    def reset(self, **kwargs):
        self.saved_shaped_reward_sum = self.shaped_reward_sum
        self.shaped_reward_sum = 0.0

        obs, info = self.env.reset(**kwargs)
        self.prev_v_x = obs[2]
        self.prev_v_y = obs[3]

        # Calculate initial distance to the landing pad (0,0)
        x_pos = obs[0]
        y_pos = obs[1]
        x_weight = 2.0
        y_weight = 1.0
        self.prev_distance = np.sqrt((x_pos * x_weight)**2 + (y_pos * y_weight)**2)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Deducting 0.05 per frame forces swift, decisive landings
        reward -= 0.05

        x_pos = obs[0]
        y_pos = obs[1]
        # v_x = obs[2]
        # v_y = obs[3]
        # angle = obs[4]
        # angle_v = obs[5]
        # L-Leg = obs[6]
        # R-Leg = obs[7]

        x_weight = 2.0
        y_weight = 1.0

        current_distance = np.sqrt((x_pos * x_weight)**2 + (y_pos * y_weight)**2)
        delta_distance = self.prev_distance - current_distance
        descent_multiplier = 100.0
        reward += (delta_distance * descent_multiplier)

        # 3. Final Evaluation: Soft Landing Bonus
        if terminated:
            impact_speed_y = abs(self.prev_v_y)
            impact_speed_x = abs(self.prev_v_x)

            # Reward vertical gentleness
            if impact_speed_y < 0.15:
                soft_landing_bonus_y = (0.15 - impact_speed_y) * 200.0
                reward += soft_landing_bonus_y

            # Reward horizontal braking
            if impact_speed_x < 0.15:
                soft_landing_bonus_x = (0.15 - impact_speed_x) * 200.0
                reward += soft_landing_bonus_x
        elif truncated:
            reward -= 100

        self.shaped_reward_sum += reward

        # Update buffers for the next frame
        self.prev_v_x = obs[2]
        self.prev_v_y = obs[3]
        self.prev_distance = current_distance

        return obs, reward, terminated, truncated, info


class StateCaptureWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_raw_obs = np.zeros(8)
        self.saved_raw_reward_sum = 0.0
        self.raw_reward_sum = 0.0
        self.raw_reward = 0.0

    def reset(self, **kwargs):
        self.saved_raw_reward_sum = self.raw_reward_sum
        self.raw_reward_sum = 0.0

        obs, info = self.env.reset(**kwargs)
        self.last_raw_obs = obs
        self.raw_reward = 0.0

        return obs, info

    def step(self, action):
        prev_raw_obs = self.last_raw_obs.copy()

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.raw_reward = reward
        self.raw_reward_sum += reward

        if terminated or truncated:
            self.last_ep_raw_reward = self.raw_reward_sum

        # --- Anti-Spike Filter for Dashboard UI ---
        if terminated:
            temp_ops = obs.copy()
            temp_ops[2] = prev_raw_obs[2]  # Freeze X-Vel
            temp_ops[3] = prev_raw_obs[3]  # Freeze Y-Vel
            temp_ops[5] = prev_raw_obs[5]  # Freeze AngVel
            self.last_raw_obs = temp_ops
        else:
            self.last_raw_obs = obs

        return obs, reward, terminated, truncated, info


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_v_y = 0.0
        self.prev_v_x = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_v_x = obs[2]
        self.prev_v_y = obs[3]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 2. Urgency Tax: Force the agent to descend by making hovering expensive
        # Deducting 0.05 per frame forces swift, decisive landings
        reward -= 0.05

        # 3. Final Evaluation: Soft Landing Bonus
        if terminated:
            impact_speed_y = abs(self.prev_v_y)
            impact_speed_x = abs(self.prev_v_x)

            # Reward vertical gentleness
            if impact_speed_y < 0.15:
                soft_landing_bonus_y = (0.15 - impact_speed_y) * 200.0
                reward += soft_landing_bonus_y

            # Reward horizontal braking
            if impact_speed_x < 0.15:
                soft_landing_bonus_x = (0.15 - impact_speed_x) * 200.0
                reward += soft_landing_bonus_x

        # Update buffers for the next frame
        self.prev_v_x = obs[2]
        self.prev_v_y = obs[3]

        return obs, reward, terminated, truncated, info


class VisionObservationWrapper(gym.ObservationWrapper):
    """
    Converts 8D state into 84x84 grayscale image.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        frame = self.env.render()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
        return resized_frame
