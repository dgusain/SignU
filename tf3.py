# upgraded the neural network immensely, increased till 5 residual blocks, increased lstm size, increased ACL layers and increased hidden units. 
# allowed training plots to be plotted regularly
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco_py
import os
import torch
from torch import nn
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
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
        self.dropout = nn.Dropout(p=0.3)  # Added dropout
        self.fc2 = nn.Linear(size, size)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout
        out = self.fc2(out)
        out += residual
        out = self.layer_norm(out)
        out = self.relu(out)
        return out


class CustomFE(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 2048):
        super(CustomFE, self).__init__(observation_space, features_dim)
        self.initial_net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)  # Added a fifth residual block
        )
        self.final_net = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.initial_net(observations)
        x = self.residual_blocks(x)
        x = self.final_net(x)
        return x
    

class CustomACLstmPolicy(MlpLstmPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomACLstmPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomFE,
            features_extractor_kwargs=dict(features_dim=2048),  # Increased feature dimension
            net_arch=dict(
                pi=[2048, 2048, 1024],  # Added an additional layer
                vf=[2048, 2048, 1024]   # Added an additional layer
            ),
            activation_fn=nn.ReLU,
            lstm_hidden_size=1024  # Increased LSTM hidden size
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
        self.ep = 1

        actuator_ranges = self.sim.model.actuator_ctrlrange
        self.actuator_min = torch.tensor(actuator_ranges[:, 0], dtype=torch.float32)
        self.actuator_max = torch.tensor(actuator_ranges[:, 1], dtype=torch.float32)

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

        self.all_actuators = (self.finger_actuators["ForeFinger"] +
                              self.finger_actuators["MiddleFinger"] +
                              self.finger_actuators["RingFinger"] +
                              self.finger_actuators["LittleFinger"] +
                              self.finger_actuators["Thumb"])

        num_actuators = len(self.all_actuators)
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
        rescaled_action = np.zeros(len(self.all_actuators))
        #print("Action: ", action)
        for i, bin_id in enumerate(action):
            fraction = bin_id / 10.0
            actuator_id = self.all_actuators[i]
            actuator_min = self.sim.model.actuator_ctrlrange[actuator_id, 0]
            actuator_max = self.sim.model.actuator_ctrlrange[actuator_id, 1]
            target_position = actuator_min + fraction * (actuator_max - actuator_min)
            rescaled_action[i] = target_position

        joint_positions_dict = {act_id: pos for act_id, pos in zip(self.all_actuators, rescaled_action)}
        self.set_joint_positions(joint_positions_dict, steps=100, delay=0.00001)

        state = torch.from_numpy(self.sim.data.qpos[:24].astype(np.float32))
        new_difference_angle = np.abs(self.compute_difference_angle(state))


        if new_difference_angle < self.target_threshold:
            reward = 5.0
            done = True
            self.ep += 1
        elif new_difference_angle < 1 and new_difference_angle > 0.5:
            reward = 2.0
            done = False
        elif new_difference_angle < 0.5 and new_difference_angle > 0.1:
            reward = 3.0
            done = False
        elif new_difference_angle < 0.1 and new_difference_angle > self.target_threshold:
            reward = 4.0
            done = False
        elif self.steps_taken >= self.max_steps:
            reward = new_difference_angle
            done = True
            self.ep += 1
        else:
            reward = 0  # the angle = reward (we are not computing the difference)
            done = False
        #print(f"Episode: {self.ep} | Step: {self.steps_taken} | Reward: {reward} | Difference angle: {new_difference_angle}")
        self.last_difference_angle = new_difference_angle
        reward += -0.001 * self.steps_taken
        self.steps_taken += 1

        truncated = False
        info = {}

        return state.numpy(), reward, done, truncated, info

    def set_joint_positions(self, joint_positions, steps=200, delay=0.02):
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
    def __init__(self, num_envs, plot_freq=100, save_path='/home/dgusain/Bot_hand/figs/03/', verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.num_envs = num_envs
        self.plot_freq = plot_freq  # Number of episodes between plot saves
        self.save_path = save_path
        self.episode_rewards = [[] for _ in range(num_envs)]
        self.episode_counts = [0 for _ in range(num_envs)]
        self.all_rewards = []  # List to store all episode rewards

        # Ensure the save_path directory exists
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        rewards = self.locals.get('rewards', [])
        
        for i, (done, reward) in enumerate(zip(dones, rewards)):
            self.episode_rewards[i].append(reward)
            if done:
                self.episode_counts[i] += 1
                total_reward = np.sum(self.episode_rewards[i])
                self.all_rewards.append(total_reward)
                avg_reward = np.mean(self.episode_rewards[i])
                max_reward = np.max(self.episode_rewards[i])
                
                print(f"Episode {self.episode_counts[i]} | Env {i+1} | "
                      f"Total Reward: {total_reward:.2f} | "
                      f"Avg. Reward: {avg_reward:.2f} | Max Reward: {max_reward:.2f}")
                
                # Reset the rewards for the environment
                self.episode_rewards[i] = []
                
                # Check if it's time to plot
                if self.episode_counts[i] % self.plot_freq == 0:
                    self.plot_rewards(self.episode_counts[i])

        return True

    def plot_rewards(self, episode):
        plt.figure(figsize=(10, 5))
        plt.plot(self.all_rewards, label="Total Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"Training Reward Progress up to Episode {episode}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plot_filename = os.path.join(self.save_path, f"training_rewards_episode_{episode}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved training plot to {plot_filename}")

    def plot_final_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.all_rewards, label="Total Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Final Training Reward Progress")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plot_filename = os.path.join(self.save_path, f"final_training_rewards.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved final training plot to {plot_filename}")

def make_env():
    def _init():
        env = HandEnv()
        return env
    return _init

def inference(gpu_id, num_envs):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = DummyVecEnv([make_env() for _ in range(num_envs)])
    env = VecMonitor(env)
    #total_timesteps = episodes * steps_per_episode * num_envs

    model_path = "/home/dgusain/Bot_hand/agents/agent_tf_03"
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"No saved model found at {model_path}.zip")

    model = RecurrentPPO.load(model_path, device=device)
    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((num_envs,), dtype=bool)
    eval_rewards = []
    for step in range(10):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts)
        obs, reward, done, truncated = env.step(action)
        #env.render() 
        print(f"Step: {step}, Action: {action}, Reward: {reward}, Done: {done}")
        eval_rewards.append(reward)
        if done[0]:
            obs = env.reset()
            lstm_states = None
            episode_starts = np.ones((num_envs,), dtype=bool)
        else:
            episode_starts = np.zeros((num_envs,), dtype=bool)

def train_on_gpu(gpu_id: int, num_envs: int, episodes: int, steps_per_episode: int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = DummyVecEnv([make_env() for _ in range(num_envs)])
    env = VecMonitor(env)
    total_timesteps = episodes * steps_per_episode * num_envs
    model = RecurrentPPO(
        CustomACLstmPolicy,
        env,
        verbose=1,
        device=device,
        ent_coef=0.01,
        learning_rate=0.0003,
        clip_range=0.2,
        n_steps=steps_per_episode, 
        batch_size=8192,
        gamma=0.998,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,
        use_sde=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  
        save_path="/home/dgusain/Bot_hand/agents/checkpoints/03/",  
        name_prefix="agent_tf"  #
    )
    plot_frequency = 100 
    save_plots_path = "/home/dgusain/Bot_hand/figs/03/"  
    reward_callback = RewardCallback(num_envs=num_envs, plot_freq=plot_frequency, save_path=save_plots_path)

    callback = CallbackList([reward_callback, checkpoint_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10)
    model.save(f"/home/dgusain/Bot_hand/agents/agent_tf3")
    reward_callback.plot_final_rewards()

    return callback


def main():
    gpu_id = 3
    num_envs = 8     # Use a single environment for simpler episode count
    episodes = 10000   # Number of episodes
    steps_per_episode = 1024  # Steps per episode

    callback = train_on_gpu(gpu_id, num_envs, episodes, steps_per_episode)

if __name__ == "__main__":
    main()
