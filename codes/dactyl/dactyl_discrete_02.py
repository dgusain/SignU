import gymnasium as gym
from gymnasium import spaces
from statistics import median
import numpy as np
import mujoco_py
import os
import torch
from torch import nn
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import logging
import matplotlib.pyplot as plt  
import math
import time

logging.basicConfig(level=logging.INFO)

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super(ResidualBlock, self).__init__()   
        self.fc1 = nn.Linear(size, size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(size, size)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual  # Residual connection
        out = self.layer_norm(out)
        out = self.relu(out)
        return out

class CustomFE(BaseFeaturesExtractor):
    """
    Custom feature extractor for HandEnv.
    Incorporates residual layers and layer normalization.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super(CustomFE, self).__init__(observation_space, features_dim)
        self.initial_net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.final_net = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.initial_net(observations)
        x = self.residual_blocks(x)
        x = self.final_net(x)
        return x

class CustomACLstmPolicy(MlpLstmPolicy):
    """
    Custom Actor-Critic Policy with LSTM and custom feature extractor.
    """

    def __init__(self, *args, **kwargs):
        super(CustomACLstmPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomFE,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=nn.ReLU,
            lstm_hidden_size=128
        )

class HandEnv(gym.Env):
    def __init__(self):
        super(HandEnv, self).__init__()
        xml_path = os.path.expanduser('/home/dgusain/Bot_hand/bot_hand.xml')
        logging.info(f"Attempting to load Mujoco model from: {xml_path}")

        if not os.path.isfile(xml_path):
            logging.error(f"XML file not found at {xml_path}.")
            raise FileNotFoundError(f"XML file not found at {xml_path}.")

        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        logging.info("Mujoco model loaded successfully.")

        self.initial_state = self.sim.get_state()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        self.target_threshold = 0.05
        self.max_steps = 100
        self.steps_taken = 0

        # Actuator ranges
        actuator_ranges = self.sim.model.actuator_ctrlrange
        self.actuator_min = torch.tensor(actuator_ranges[:, 0], dtype=torch.float32)
        self.actuator_max = torch.tensor(actuator_ranges[:, 1], dtype=torch.float32)
        
        # Define actuator mapping based on actuator names from the model
        self.finger_actuators = {
            "ForeFinger": [self.sim.model.actuator_name2id(name) for name in [
                "ForeFingerJoint0_act", "ForeFingerJoint1_act", "ForeFingerJoint2_act", "ForeFingerJoint3_act"]],
            "MiddleFinger": [self.sim.model.actuator_name2id(name) for name in [
                "MiddleFingerJoint0_act", "MiddleFingerJoint1_act", "MiddleFingerJoint2_act", "MiddleFingerJoint3_act"]],
            "RingFinger": [self.sim.model.actuator_name2id(name) for name in [
                "RingFingerJoint0_act", "RingFingerJoint1_act", "RingFingerJoint2_act", "RingFingerJoint3_act"]],
            "LittleFinger": [self.sim.model.actuator_name2id(name) for name in [
                "LittleFingerJoint0_act", "LittleFingerJoint1_act", "LittleFingerJoint2_act", "LittleFingerJoint3_act"]],
            "Thumb": [self.sim.model.actuator_name2id(name) for name in [
                "ThumbJoint0_act", "ThumbJoint1_act", "ThumbJoint2_act", "ThumbJoint3_act", "ThumbJoint4_act"]],
        }

        # Flatten all actuators into a single list in a known order
        self.all_actuators = (self.finger_actuators["ForeFinger"] +
                              self.finger_actuators["MiddleFinger"] +
                              self.finger_actuators["RingFinger"] +
                              self.finger_actuators["LittleFinger"] +
                              self.finger_actuators["Thumb"])

        num_actuators = len(self.all_actuators)
        # MultiDiscrete action space: 11 bins per actuator
        self.action_space = spaces.MultiDiscrete([11]*num_actuators)

        self.ground_truth_quats = torch.tensor(self.get_ground_truth_quaternion(), dtype=torch.float32)
        self.last_difference_angle = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        self.steps_taken = 0
        obs = torch.from_numpy(self.sim.data.qpos[:24].astype(np.float32))
        self.last_difference_angle = self.compute_difference_angle(obs)
        return obs.numpy(), {}

    def step(self, action: np.ndarray):
        # action[i] in [0..10]
        # Map discrete action to continuous target
        rescaled_action = np.zeros(len(self.all_actuators))
        for i, bin_id in enumerate(action):
            fraction = bin_id / 10.0
            actuator_id = self.all_actuators[i]
            actuator_min = self.sim.model.actuator_ctrlrange[actuator_id, 0]
            actuator_max = self.sim.model.actuator_ctrlrange[actuator_id, 1]
            target_position = actuator_min + fraction * (actuator_max - actuator_min)
            rescaled_action[i] = target_position

        # Set joint positions directly
        joint_positions_dict = {act_id: pos for act_id, pos in zip(self.all_actuators, rescaled_action)}
        self.set_joint_positions(joint_positions_dict, steps=1, delay=0.0)

        state = torch.from_numpy(self.sim.data.qpos[:24].astype(np.float32))
        new_difference_angle = self.compute_difference_angle(state)

        # Compute reward and done
        if new_difference_angle < self.target_threshold:
            reward = 5.0
            done = True
        elif self.steps_taken >= self.max_steps:
            reward = -20.0
            done = True
        else:
            # Reward = d(t) - d(t+1)
            reward = self.last_difference_angle - new_difference_angle
            done = False

        self.last_difference_angle = new_difference_angle
        self.steps_taken += 1

        truncated = False
        info = {}

        return state.numpy(), reward, done, truncated, info

    def set_joint_positions(self, joint_positions, steps=1, delay=0.0):
        """
        Set joint positions. For RL, steps=1 is typically fine.
        """
        for i in range(steps):
            for joint_id, target_position in joint_positions.items():
                current_position = self.sim.data.ctrl[joint_id]
                new_position = current_position + (target_position - current_position) * ((i+1) / steps)
                self.sim.data.ctrl[joint_id] = new_position
            self.sim.step()
            if delay > 0:
                time.sleep(delay)

    def compute_difference_angle(self, state: torch.Tensor) -> float:
        rendered_quat = self.get_rendered_pose_quaternion()
        rendered_quat = self.normalize_quaternion(rendered_quat)
        gt_quat = self.normalize_quaternion(self.ground_truth_quats)

        dot_product = torch.sum(rendered_quat * gt_quat, dim=1)
        angle_per_joint = 2 * torch.acos(torch.clamp(dot_product, -1.0, 1.0))
        avg_angle = angle_per_joint.mean().item()
        return avg_angle

    def get_rendered_pose_quaternion(self) -> torch.Tensor:
        quaternions = []
        for joint in range(self.model.njnt):
            q = self.sim.data.qpos[self.model.jnt_qposadr[joint]:self.model.jnt_qposadr[joint] + 4]
            quaternions.append(q)
        quaternions_np = np.array(quaternions, dtype=np.float32)
        return torch.from_numpy(quaternions_np)

    def normalize_quaternion(self, q: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(q, dim=1, keepdim=True)
        return q / norm.clamp(min=1e-8)

    def get_ground_truth_quaternion(self) -> np.ndarray:
        ground_truth_quats = [
            [6.32658406e-04,  1.25473697e-03, -9.99718591e-03,  1.56702220e+00],
            [1.25473697e-03, -9.99718591e-03,  1.56702220e+00,  1.56730258e+00],
            [-0.00999719,  1.5670222,   1.56730258,  1.5674143],
            [1.5670222,   1.56730258,  1.5674143,  -0.00999801],
            [1.56730258,  1.5674143,  -0.00999801,  1.56701926],
            [1.5674143,  -0.00999801,  1.56701926,  1.56730194],
            [-0.00999801,  1.56701926,  1.56730194,  1.5674143],
            [1.56701926,  1.56730194,  1.5674143,  -0.00999756],
            [1.56730194,  1.5674143,  -0.00999756,  1.56701717],
            [1.5674143,  -0.00999756,  1.56701717,  1.56730177],
            [-0.00999756,  1.56701717,  1.56730177,  1.56741429],
            [1.56701717, 1.56730177, 1.56741429, 0.00868252],
            [1.56730177,  1.56741429,  0.00868252, -0.01000805],
            [1.56741429,  0.00868252, -0.01000805,  1.56708349],
            [0.00868252, -0.01000805,  1.56708349,  1.56730911],
            [-0.01000805,  1.56708349,  1.56730911,  1.56741508],
            [1.56708349, 1.56730911, 1.56741508, 0.49916711],
            [1.56730911, 1.56741508, 0.49916711, 0.50022545],
            [1.56741508, 0.49916711, 0.50022545, 0.25330455],
            [4.99167109e-01, 5.00225452e-01, 2.53304550e-01, 4.08440608e-04],
            [5.00225452e-01,  2.53304550e-01,  4.08440608e-04, -9.00000689e-01],
            [2.53304550e-01,  4.08440608e-04, -9.00000689e-01,  1.00000000e-01],
            [4.08440608e-04, -9.00000689e-01,  1.00000000e-01, -1.00000000e-01],
            [-0.90000069,  0.1,        -0.1,         0.01463168],
            [0.1,        -0.1,         0.01463168,  1.0]
        ]
        return np.array(ground_truth_quats, dtype=np.float32)

class RewardCallback(BaseCallback):
    def __init__(self, avg_interval=100):
        super(RewardCallback, self).__init__()
        self.episode_num = 0
        self.max_reward = -np.inf
        self.avg_interval = avg_interval
        self.rewards_list = []
        self.median_rewards = {}
        self.avg_rewards = {}
        self.iteration_num = 0     

    def _on_rollout_end(self) -> bool:
        self.iteration_num += 1
        return True

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        rewards = self.locals.get('rewards', [])
        if np.any(dones):
            for done, reward in zip(dones, rewards):
                if done:
                    self.episode_num += 1
                    if reward > self.max_reward:
                        self.max_reward = reward
                    self.rewards_list.append(reward)
                    if len(self.rewards_list) == self.avg_interval:
                        median_reward = median(self.rewards_list)
                        avg_reward = np.mean(self.rewards_list)
                        block_num = self.episode_num // self.avg_interval
                        self.median_rewards[block_num] = median_reward
                        self.avg_rewards[block_num] = avg_reward
                        print(f"Iteration: {self.iteration_num} | Episodes: {block_num * self.avg_interval} | Median Reward: {median_reward:.2f} | Avg Reward: {avg_reward:.2f} | Max Reward: {self.max_reward:.2f}")
                        self.rewards_list = []
        return True

    def _on_training_end(self) -> None:
        if len(self.rewards_list) > 0:
            median_reward = median(self.rewards_list)
            avg_reward = np.mean(self.rewards_list)
            block_num = self.episode_num // self.avg_interval + 1
            self.median_rewards[block_num] = median_reward
            self.avg_rewards[block_num] = avg_reward
            current_max = max(self.rewards_list)
            if current_max > self.max_reward:
                self.max_reward = current_max           
            print(f"Training End | Episodes: {self.episode_num} | Median Reward: {median_reward:.2f} | Avg Reward: {avg_reward:.2f} | Max Reward: {self.max_reward:.2f}")
            self.rewards_list = []

def make_env():
    def _init():
        env = HandEnv()
        return env
    return _init

def train_on_gpu(gpu_id: int, num_envs: int, total_timesteps: int, num_steps: int, save_interval: int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = DummyVecEnv([make_env() for _ in range(num_envs)])  
    env = VecMonitor(env)

    model = RecurrentPPO(
        CustomACLstmPolicy,  
        env,
        verbose=1,
        device=device,
        ent_coef=0.2,
        learning_rate=0.0005,
        clip_range=0.3,
        n_steps=num_steps,
        batch_size=4096,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,
        use_sde=False,
    )

    callback = RewardCallback(avg_interval=100)
    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10)

    model.save(f"/home/dgusain/Bot_hand/agents/agent_code_discrete02_gpu{gpu_id}")
    return callback

def main():
    gpu_id = 2
    num_envs = 4
    n_iter = 1500
    n_steps = 1024
    total_timesteps = n_iter * n_steps * num_envs
    save_interval = 25

    callback = train_on_gpu(gpu_id, num_envs, total_timesteps, n_steps, save_interval)
    block_numbers = list(callback.median_rewards.keys())
    block_median_rewards = list(callback.median_rewards.values())
    block_avg_rewards = list(callback.avg_rewards.values())
    episode_numbers = [block_num * 100 for block_num in block_numbers]

    plt.figure(figsize=(10, 6))
    plt.plot(episode_numbers, block_median_rewards, marker='o', linestyle='-', color='b', label='Median Reward')
    plt.plot(episode_numbers, block_avg_rewards, marker='x', linestyle='--', color='orange', label='Average Reward')
    plt.xlabel('Episode Number')
    plt.ylabel('Reward (per 100 episodes)')
    plt.title('Reward vs Episode Number')
    plt.grid(True)
    plt.savefig('/home/dgusain/Bot_hand/figs/fig_code_discrete02.png')
    plt.close() 
    print("Plot saved as fig_code_discrete02.png")

if __name__ == "__main__":
    main()
