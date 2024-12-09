import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco_py
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import openai
import wandb

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

###############################################################
# GPT Query Function
###############################################################
def query_gpt_for_digit(joint_positions):
    """
    Query GPT-4 to interpret the given joint positions (angles) as a digit (1-5).
    Joint positions is a numpy array of joint angles.

    The prompt is minimal. In practice, improve prompt engineering.
    """
    prompt = f"""
You are given the joint angles of a Shadow Dexterous Hand as a list of floats. They represent a hand gesture. 
Your job is to determine which digit from 1 to 5 the hand is representing by how many fingers are extended.

Joint positions: {joint_positions.tolist()}

Please respond with a single digit (1,2,3,4, or 5) that best matches the gesture.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs only the digit."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        guess = response.choices[0].message.content.strip()
        guess = int(guess)  # Convert to integer
        if guess not in [1,2,3,4,5]:
            guess = 0  # If invalid, set to 0
    except:
        guess = 0
    return guess


###############################################################
# Custom Environment for GPT-based Reward
###############################################################
class GPTDigitHandEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, xml_path, target_digit=3, max_steps=50):
        super(GPTDigitHandEnv, self).__init__()
        self.target_digit = target_digit
        self.max_steps = max_steps
        self.steps_taken = 0

        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"XML file not found at {xml_path}")
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)

        # Let's assume number of actuators and joint observation size from model
        num_actuators = self.model.nu
        num_qpos = self.model.nq
        num_qvel = self.model.nv

        # Observation: qpos + qvel
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_qpos + num_qvel,), dtype=np.float32
        )

        # Action: actuator controls
        actuator_ranges = self.sim.model.actuator_ctrlrange
        self.actuator_min = actuator_ranges[:, 0]
        self.actuator_max = actuator_ranges[:, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_actuators,), dtype=np.float32)

        self.initial_state = self.sim.get_state()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        self.steps_taken = 0
        return self._get_obs(), {}

    def step(self, action):
        # Rescale action
        rescaled_action = self.actuator_min + (action + 1) * (self.actuator_max - self.actuator_min) / 2
        rescaled_action = np.clip(rescaled_action, self.actuator_min, self.actuator_max)
        self.sim.data.ctrl[:] = rescaled_action
        self.sim.step()
        obs = self._get_obs()

        # Query GPT for digit guess
        qpos = obs[:self.model.nq]  # positions
        guess = query_gpt_for_digit(qpos)

        # Reward and done
        if guess == self.target_digit:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False

        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            done = True

        # truncated in gymnasium means time limit, here we can just say truncated = False
        truncated = False
        info = {}
        return obs, reward, done, truncated, info

    def _get_obs(self):
        qpos = self.sim.data.qpos[:].astype(np.float32)
        qvel = self.sim.data.qvel[:].astype(np.float32)
        return np.concatenate([qpos, qvel])

    def render(self, mode='human'):
        # Implement a visualization if desired
        pass

    def close(self):
        pass


###############################################################
# Example Training Code
###############################################################
# If you want to integrate W&B, callbacks, etc., similar to your old code:
class SimpleRewardCallback(BaseCallback):
    def __init__(self, avg_interval=10, verbose=0):
        super(SimpleRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.avg_interval = avg_interval

    def _on_step(self):
        # Check if any episode finished
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        rewards = self.locals.get('rewards', [])

        for done, info, reward in zip(dones, infos, rewards):
            if done:
                self.episode_rewards.append(reward)
                if len(self.episode_rewards) % self.avg_interval == 0:
                    avg_reward = np.mean(self.episode_rewards[-self.avg_interval:])
                    print(f"Episode {len(self.episode_rewards)}: Avg Reward (last {self.avg_interval} eps): {avg_reward:.2f}")
                    wandb.log({"avg_reward": avg_reward}, step=self.num_timesteps)
        return True

def main():
    # Initialize W&B if desired
    wandb.init(project="shadow_hand_gpt_digit", name="gpt_digit_control")

    # Create Env
    xml_path = "/path/to/shadow_hand.xml"  # Update with your hand model path
    env = GPTDigitHandEnv(xml_path=xml_path, target_digit=3, max_steps=50)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env)

    callback = SimpleRewardCallback(avg_interval=10)

    # Use a simple MlpPolicy
    model = PPO("MlpPolicy", env, verbose=1, device="cuda", n_steps=2048, batch_size=64, learning_rate=3e-4)
    model.learn(total_timesteps=100000, callback=callback)

    model.save("gpt_digit_hand_model.zip")
    wandb.finish()

if __name__ == "__main__":
    main()
