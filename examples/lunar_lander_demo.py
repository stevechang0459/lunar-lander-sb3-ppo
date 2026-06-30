import gymnasium as gym

# 1. Initialize the LunarLander environment
# "human" mode allows us to watch the lander interact with the physics engine live
env = gym.make("LunarLander-v3", render_mode="human")

# 2. Reset the environment to begin the first episode
# It returns the initial state observation and a dictionary with diagnostic info
observation, info = env.reset(seed=42)

print("Initial Observation Vector:")
print(observation)

# 3. Main simulation loop
for step in range(300):

    # Sample a random action from the discrete action space (0, 1, 2, or 3)
    # 0: Do nothing, 1: Fire left orientation engine
    # 2: Fire main engine, 3: Fire right orientation engine
    action = env.action_space.sample()

    # Apply the action to the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # 4. Check if the lander has crashed, landed safely, or timed out
    if terminated or truncated:
        print(f"Episode finished after {step} steps. Resetting...")
        observation, info = env.reset()

# 5. Safely close the environment window and release memory
env.close()
