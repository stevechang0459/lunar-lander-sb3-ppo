# Lunar Lander - Stable-Baselines3 PPO

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Stable Baselines3](https://img.shields.io/badge/Stable%20Baselines3-PPO-purple.svg)](https://stable-baselines3.readthedocs.io/)

This repository implements a Reinforcement Learning agent to solve the `LunarLander-v3` environment using the Proximal Policy Optimization (PPO) algorithm provided by Stable-Baselines3.

<div align="center">
  <img width="451" height="324" alt="Lunar Lander Gameplay Screenshot 1" src="https://github.com/user-attachments/assets/d4dde00a-2153-4c22-9c4b-f4c5f594dfca" />
  <img width="451" height="324" alt="Lunar Lander Gameplay Screenshot 2" src="https://github.com/user-attachments/assets/c1093a2e-8cac-463f-a00c-9ac015596930" />
</div>

## Project Structure

Brief overview of the main scripts included in this repository:

- `train_mlp.py`: Trains the PPO agent using a single Gymnasium environment.
- `train_mlp_multi_env.py`: Trains the agent using vectorized environments (SubprocVecEnv) for faster data collection and training.
- `test_mlp.py`: Loads the trained model and visualizes its performance in the environment.
- `luner_lander_v3.py`: Base environment setup or testing script.

## Installation

It is recommended to use an Anaconda environment to manage dependencies.

```shell
conda create -n lunar-lander python=3.11
conda activate lunar-lander

# Install core dependencies
pip install swig
pip install gymnasium
pip install gymnasium[box2d]
pip install gymnasium[other]
pip install stable-baselines3
pip install tensorboard
```

## Usage

### 1. Training the Agent

You can choose between single-environment training or multi-environment training (recommended for PPO).

```shell
# Standard training on a single environment
python train_mlp.py

# Accelerated training using multiple environments
python train_mlp_multi_env.py
```

### 2. Testing the Model

To watch the trained agent land the spacecraft, run the test script:

```shell
python test_mlp.py
```

### 3. Monitoring Training Logs

Training metrics (e.g., episode reward, policy entropy, value loss) are logged via TensorBoard. To view them in your browser:

```shell
tensorboard --logdir ./tb_logs/
```

Then navigate to http://localhost:6006 in your web browser.

## Results

After training for **20,000,768 timesteps**, the PPO agent achieved an average raw reward of **131**. The model demonstrates an excellent understanding of the environment, indicated by a high explained variance of **0.943** in the value function.

**Training Highlights:**
- **Total Timesteps:** 20,000,768
- **Average Raw Reward:** 131 (`ep_rew_mean`)
- **Average Shaped Reward:** 143 (`shaped_reward_mean`)
- **Training Time:** ~60 minutes (3,625 seconds)
- **Throughput:** 5,516 FPS

<details>
<summary><b>Click to expand the full TensorBoard training log</b></summary>

```text
-------------------------------------------
| custom/                 |               |
|    1_raw_reward_mean    | 131           |
|    2_shaped_reward_mean | 143           |
|    3_steps_mean         | 782           |
|    4_n_steps            | 20000768      |
| rollout/                |               |
|    ep_len_mean          | 782           |
|    ep_rew_mean          | 131           |
| time/                   |               |
|    fps                  | 5516          |
|    iterations           | 4883          |
|    time_elapsed         | 3625          |
|    total_timesteps      | 20000768      |
| train/                  |               |
|    approx_kl            | 3.3342803e-07 |
|    clip_fraction        | 0             |
|    clip_range           | 0.025         |
|    entropy_loss         | -1.01         |
|    explained_variance   | 0.943         |
|    learning_rate        | 2.54e-06      |
|    loss                 | 65.3          |
|    n_updates            | 19528         |
|    policy_gradient_loss | -4.23e-06     |
|    value_loss           | 79.2          |
-------------------------------------------
```
</details>

## References

- [Gymnasium Box2D Environments](https://gymnasium.farama.org/environments/box2d/)
- [Lunar Lander Environment Details](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable-Baselines3 Repository](https://github.com/DLR-RM/stable-baselines3)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
