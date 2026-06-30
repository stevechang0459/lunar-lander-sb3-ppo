# train_mlp.py
import os
import time
import datetime
import random
import numpy as np
from typing import Callable, Union
from collections import deque
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from lunar_lander_wrappers import MlpStateCaptureWrapper, MlpCustomRewardWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor


N_ENVS = 1
TOTAL_TIMESTEPS = 10000000
N_STEPS = 4096 // N_ENVS
BATCH_SIZE = 512
N_EPOCHS = 4
GAMMA = 0.94
ENT_COEF = 0.02
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
    def __init__(self, verbose=0):
        super().__init__(verbose)

        # Keep track of the last 100 episodes for moving averages
        self.raw_reward_history = deque(maxlen=100)
        self.shaped_reward_history = deque(maxlen=100)
        self.step_count_history = deque(maxlen=100)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        done = self.locals["dones"][0]
        if done:
            saved_raw_reward_sum = env.get_wrapper_attr("saved_raw_reward_sum")
            saved_shaped_reward_sum = env.get_wrapper_attr("saved_shaped_reward_sum")
            saved_ep_step_count = env.get_wrapper_attr("saved_ep_step_count")
            # 4. Update histories for moving averages
            self.raw_reward_history.append(saved_raw_reward_sum)
            self.shaped_reward_history.append(saved_shaped_reward_sum)
            self.step_count_history.append(saved_ep_step_count)

        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("custom/1_raw_reward_mean", np.mean(self.raw_reward_history))
        self.logger.record("custom/2_shaped_reward_mean", np.mean(self.shaped_reward_history))
        self.logger.record("custom/3_steps_mean", np.mean(self.step_count_history))
        self.logger.record("custom/4_n_steps", self.num_timesteps)


if __name__ == "__main__":
    BASE_SEED = int(time.time() * 1000) % 100000
    set_random_seed(BASE_SEED)
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    # torch.manual_seed(BASE_SEED)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(BASE_SEED)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_PREFIX = f"ppo_mlp_{current_time}"

    # Create a specific directory for this training run
    base_dir = "models_cnn"
    model_dir = os.path.join(base_dir, MODEL_PREFIX)
    os.makedirs(model_dir, exist_ok=True)

    # Build environment
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = Monitor(env)    # monitor raw reward
    env_state = MlpStateCaptureWrapper(env)
    env = MlpCustomRewardWrapper(env_state)
    # env = Monitor(env)    # monitor shaped reward

    LOG_DIR = "./tb_logs/"
    model = PPO(
        policy="MlpPolicy",
        env=env,
        device="cpu",
        verbose=1,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        learning_rate=linear_schedule(INITIAL_LR, FINAL_LR),
        clip_range=linear_schedule(INITIAL_CR, FINAL_CR),
        tensorboard_log=LOG_DIR
    )

    # Save checkpoints inside the model directory
    checkpoint_callback = CheckpointCallback(
        save_freq=TOTAL_TIMESTEPS//20,
        save_path=model_dir,
        name_prefix=MODEL_PREFIX
    )

    # Bundle them together
    logging_callback = LoggingCallback()
    callback_list = CallbackList([checkpoint_callback, logging_callback])

    print(f"Starting SB3 PPO training: {MODEL_PREFIX}")
    print(f"Models will be saved to: {model_dir}")
    print(f"TensorBoard log directory: {os.path.join(LOG_DIR, MODEL_PREFIX)}")

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
        env_state.close()
