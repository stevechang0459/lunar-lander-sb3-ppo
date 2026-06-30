# train_cnn.py
import os
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from typing import Callable, Union
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
import datetime
from gymnasium.wrappers import FrameStackObservation
import numpy as np
from collections import deque
from lunar_lander_wrappers import StateCaptureWrapper, VisionObservationWrapper, CustomRewardWrapper


N_STEPS = 4096
BATCH_SIZE = 512 * 8
N_EPOCHS = 4
GAMMA = 0.94
ENT_COEF = 0.01
INITIAL_LR = 2.5e-4
FINAL_LR = 2.5e-6
INITIAL_CR = 0.150
FINAL_CR = 0.025


def linear_schedule(initial_value: Union[float, str], final_value: Union[float, str] = 0.0) -> Callable[[float], float]:
    """
    Creates a linear learning rate or clip range schedule.

    The progress_remaining parameter provided by SB3 decreases linearly from 1.0 to 0.0.
    This schedule smoothly interpolates between initial_value and final_value.

    Args:
        initial_value: The starting value at the beginning of training (progress_remaining = 1.0).
        final_value: The minimum value at the end of training (progress_remaining = 0.0).

    Returns:
        A function that computes the current value based on progress_remaining.
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    if isinstance(final_value, str):
        final_value = float(final_value)

    assert initial_value > 0.0, "Initial value must be greater than 0.0"

    def scheduler(progress_remaining: float) -> float:
        # Calculate the current value using linear interpolation
        return final_value + progress_remaining * (initial_value - final_value)

    return scheduler

class LoggingCallback(BaseCallback):
    """
    Hooks into the training loop to extract raw physics data from the StateCaptureWrapper
    and log it directly to TensorBoard. Now includes a 100-episode moving average
    for both raw score and step count to align with SB3's ep_rew_mean.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_raw_reward = 0.0
        self.ep_step_count = 0
        self.total_steps = 0

        # Keep track of the last 100 episodes for moving averages
        self.raw_reward_history = deque(maxlen=100)
        self.step_count_history = deque(maxlen=100)

    def _on_step(self) -> bool:
        # 1. Get the underlying environment instance from SB3's DummyVecEnv
        env = self.training_env.envs[0]

        # 2. Accumulate raw reward and steps for the current episode
        # raw_reward = env.get_wrapper_attr("raw_reward")
        # self.ep_raw_reward += raw_reward
        self.ep_step_count += 1
        self.total_steps += 1

        # 3. Check if the episode just finished
        done = self.locals["dones"][0]
        if done:
            saved_raw_reward_sum = env.get_wrapper_attr("saved_raw_reward_sum")
            # 4. Update histories for moving averages
            self.raw_reward_history.append(saved_raw_reward_sum)
            self.step_count_history.append(self.ep_step_count)

            # Calculate the moving averages
            raw_reward_mean = np.mean(self.raw_reward_history)
            step_count_mean = np.mean(self.step_count_history)

            # 5. Push data to TensorBoard
            self.logger.record("custom/1_raw_reward_mean", raw_reward_mean)
            self.logger.record("custom/2_steps_mean", step_count_mean)
            self.logger.record("custom/3_n_steps", self.total_steps)

            # 6. Reset accumulators for the next episode
            self.ep_raw_reward = 0.0
            self.ep_step_count = 0

        return True

if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_PREFIX = f"ppo_cnn_{current_time}"
    TOTAL_TIMESTEPS = 500000 * 20

    # Create a specific directory for this training run
    base_dir = "models_cnn"
    model_dir = os.path.join(base_dir, MODEL_PREFIX)
    os.makedirs(model_dir, exist_ok=True)

    # Build environment chain
    raw_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env_state = StateCaptureWrapper(raw_env)                        # 1. 攔截 8 維狀態與燃料費
    env_reward_shaped = CustomRewardWrapper(env_state)              # 2. 塑形分數
    env_vision = VisionObservationWrapper(env_reward_shaped)        # 3. 轉換為 84x84 灰階影像
    env_stacked = FrameStackObservation(env_vision, stack_size=4)   # 4. 堆疊 4 幀

    LOG_DIR = "./tb_logs/"
    model = PPO(
        policy="CnnPolicy",
        env=env_stacked,
        device="xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cuda",
        verbose=1,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=4,
        gamma=0.94,
        ent_coef=ENT_COEF,                                     # Added Entropy Coefficient
        learning_rate=linear_schedule(INITIAL_LR, FINAL_LR),  # Updated LR schedule
        clip_range=linear_schedule(INITIAL_CR, FINAL_CR),     # Updated CR schedule
        tensorboard_log=LOG_DIR
    )

    # Save checkpoints inside the model directory
    checkpoint_callback = CheckpointCallback(
        save_freq=int(TOTAL_TIMESTEPS/20),
        save_path=model_dir,
        name_prefix=MODEL_PREFIX
    )
    logging_callback = LoggingCallback()

    # Bundle them together
    callback_list = CallbackList([checkpoint_callback, logging_callback])

    print(f"Starting SB3 PPO training: {MODEL_PREFIX}")
    print(f"Models will be saved to: {model_dir}")
    try:
        # Pass the callback_list to model.learn
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list, tb_log_name=MODEL_PREFIX)
    except KeyboardInterrupt:
        print("\n[Interrupt] Terminating early...")
    finally:
        # Save final model inside the model directory
        file_path = os.path.join(model_dir, f"{MODEL_PREFIX}_final_{TOTAL_TIMESTEPS}_steps")
        model.save(file_path)
        print(f"Model saved at {file_path}!")
        env_stacked.close()
