import gymnasium as gym
from stable_baselines3 import PPO
from shadow_hand_env import ShadowHandEnv

# Create environment
env = ShadowHandEnv(model_path="shadow_hand.xml")

# Create RL model
model = PPO("MlpPolicy", env, verbose=1)

# Train model
model.learn(total_timesteps=100000)

# Save model
model.save("shadow_hand_rl")
