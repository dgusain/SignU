# each episode contains a single step. This is a trial version to see what happens now. 
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco_py
import os
import torch
from torch import nn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecEnvWrapper, DummyVecEnv
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
import logging
import matplotlib.pyplot as plt  
import math
import time
from multiprocessing import cpu_count
import pickle
logging.basicConfig(level=logging.INFO)

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super(ResidualBlock, self).__init__()   
        self.fc1 = nn.Linear(size, size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(size, size)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
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
            nn.ReLU(inplace=True),
            nn.LayerNorm(256)
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        self.final_net = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.ReLU(inplace=True),
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
            features_extractor_kwargs=dict(features_dim=2048),
            net_arch=dict(
                pi=[2048, 2048, 1024],
                vf=[2048, 2048, 1024]
            ),
            activation_fn=nn.ReLU,
            lstm_hidden_size=1024
        )
# only used for inference
class GestureSelectorWrapper(VecEnvWrapper):
    def __init__(self, venv, gestures):
        super(GestureSelectorWrapper, self).__init__(venv)
        self.gestures = gestures
        self.num_gestures = len(gestures)
        self.gesture_indices = {gesture: idx for idx, gesture in enumerate(gestures)}
    
    def reset(self, **kwargs):
        # Expecting 'gesture' in kwargs
        gesture = kwargs.get('gesture', None)
        if gesture is not None:
            for env in self.venv.envs:
                env.current_gesture = gesture
                env.ground_truth_quats_current = np.array(env.ground_truth_quats_dict[gesture], dtype=np.float32)
        return self.venv.reset()

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
        # extracting gesture
        with open("quat_dict.pkl", 'rb') as file:
            self.ground_truth_quats_dict = pickle.load(file)  # { "a": [[q1], [q2], ...], "c": [[q1], [q2], ...], ... }
        self.gesture_names = list(self.ground_truth_quats_dict.keys())
        self.num_gestures = len(self.gesture_names)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24*4+self.num_gestures,), dtype=np.float32) # 4 quats per actuator 
        self.target_threshold = 0.04
        self.max_steps = 1  # Each episode consists of a single step
        self.steps_taken = 0
        self.ep = 1
        self.noise_std = 0.05

        actuator_ranges = self.sim.model.actuator_ctrlrange
        self.actuator_min = actuator_ranges[:, 0]
        self.actuator_max = actuator_ranges[:, 1]

        finger_names = {
            "ForeFinger": ["ForeFingerJoint0_act", "ForeFingerJoint1_act", "ForeFingerJoint2_act", "ForeFingerJoint3_act"],
            "MiddleFinger": ["MiddleFingerJoint0_act", "MiddleFingerJoint1_act", "MiddleFingerJoint2_act", "MiddleFingerJoint3_act"],
            "RingFinger": ["RingFingerJoint0_act", "RingFingerJoint1_act", "RingFingerJoint2_act", "RingFingerJoint3_act"],
            "LittleFinger": ["LittleFingerJoint0_act", "LittleFingerJoint1_act", "LittleFingerJoint2_act", "LittleFingerJoint3_act"],
            "Thumb": ["ThumbJoint0_act", "ThumbJoint1_act", "ThumbJoint2_act", "ThumbJoint3_act", "ThumbJoint4_act"],
        }

        self.finger_actuators = {
            finger: [self.sim.model.actuator_name2id(name) for name in names]
            for finger, names in finger_names.items()
        }

        self.all_actuators = np.concatenate([
            self.finger_actuators["ForeFinger"],
            self.finger_actuators["MiddleFinger"],
            self.finger_actuators["RingFinger"],
            self.finger_actuators["LittleFinger"],
            self.finger_actuators["Thumb"]
        ])

        num_actuators = len(self.all_actuators)
        self.action_space = spaces.MultiDiscrete([11]*num_actuators)
        self.current_gesture = None
        self.ground_truth_quats_current = None
        self.last_difference_angle = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #self.sim.set_state(self.initial_state)
        self.sim.reset()
        self.sim.forward()
        self.steps_taken = 0
        # Check if gesture is specified in options
        if options is not None and 'gesture' in options:
            gesture = options['gesture']
            if gesture in self.gesture_names:
                self.current_gesture = gesture
                self.ground_truth_quats_current = np.array(self.ground_truth_quats_dict[self.current_gesture], dtype=np.float32)
            else:
                raise ValueError(f"Gesture '{gesture}' not recognized.")
        else:
            # Select a random gesture
            self.current_gesture = np.random.choice(self.gesture_names)
            self.ground_truth_quats_current = np.array(self.ground_truth_quats_dict[self.current_gesture], dtype=np.float32)

        obs_quats = np.array([
            self.sim.data.qpos[self.model.jnt_qposadr[joint]:self.model.jnt_qposadr[joint] + 4]
            for joint in range(min(self.model.njnt,24))
        ]).astype(np.float32)

        # Getting current state
        obs_quaternions = np.concatenate([
            self.sim.data.qpos[self.model.jnt_qposadr[joint]:self.model.jnt_qposadr[joint] + 4]
            for joint in range(min(self.model.njnt,24))
        ]).astype(np.float32)  # Shape: (24*4,)

        # Creating a one-hot encoding for the gesture
        gesture_one_hot = np.zeros(self.num_gestures, dtype=np.float32)
        gesture_index = self.gesture_names.index(self.current_gesture)
        gesture_one_hot[gesture_index] = 1.0
        obs = np.concatenate([obs_quaternions, gesture_one_hot])

        #self.last_difference_angle = self.compute_difference_angle(obs_quats)
        return obs, {}

    def step(self, action: np.ndarray):
        # action[i] in [0..10]
        rescaled_action = np.zeros(len(self.all_actuators))
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=rescaled_action.shape)
        #print("Action: ", action)
        for i, bin_id in enumerate(action):
            fraction = bin_id / 10.0
            actuator_id = self.all_actuators[i]
            actuator_min = self.sim.model.actuator_ctrlrange[actuator_id, 0]
            actuator_max = self.sim.model.actuator_ctrlrange[actuator_id, 1]
            target_position = actuator_min + fraction * (actuator_max - actuator_min)
            rescaled_action[i] = target_position + noise[i]

        
        #noisy_action = rescaled_action + noise
        #noisy_action = np.clip(noisy_action, self.actuator_min, self.actuator_max)

        joint_positions_dict = {act_id: pos for act_id, pos in zip(self.all_actuators, rescaled_action)}
        self.set_joint_positions(joint_positions_dict, steps=200, delay=0.0)

        obs_quats = np.array([
            self.sim.data.qpos[self.model.jnt_qposadr[joint]:self.model.jnt_qposadr[joint] + 4]
            for joint in range(min(self.model.njnt,24))
        ]).astype(np.float32)  # Shape: (24*4,)

        obs_quaternions = np.concatenate([
            self.sim.data.qpos[self.model.jnt_qposadr[joint]:self.model.jnt_qposadr[joint] + 4]
            for joint in range(min(self.model.njnt,24))
        ]).astype(np.float32)
        # Creating a one-hot encoding for the gesture
        gesture_one_hot = np.zeros(self.num_gestures, dtype=np.float32)
        gesture_index = self.gesture_names.index(self.current_gesture)
        gesture_one_hot[gesture_index] = 1.0
        obs = np.concatenate([obs_quaternions, gesture_one_hot])

        new_difference_angle = np.abs(self.compute_difference_angle(obs_quats))

        # Since each episode is a single step, set done=True unconditionally
        reward = self.calculate_reward(new_difference_angle)
        done = True  # End the episode after a single step
        self.ep += 1

        self.last_difference_angle = new_difference_angle
        reward -= 0.0001 * self.steps_taken
        self.steps_taken += 1

        truncated = False
        info = {'gesture': self.current_gesture}  
        return obs, reward, done, truncated, info

    def set_joint_positions(self, joint_positions, steps=200, delay=0.02):
        for i in range(steps):
            for joint_id, target_position in joint_positions.items():
                current_position = self.sim.data.ctrl[joint_id]
                new_position = current_position + (target_position - current_position) * ((i+1) / steps)
                self.sim.data.ctrl[joint_id] = new_position
            self.sim.step()
            if delay > 0:
                time.sleep(delay)

    def calculate_reward(self, difference_angle: float) -> float:
        reward = max(-5.0, 5.0 - difference_angle * 5)  # Scales from 5.0 to 0.0
        #reward -= 0.0001 * self.steps_taken
        return reward

    def compute_difference_angle(self, state: np.ndarray) -> float:
        rendered_quat = self.normalize_quaternion(state)
        gt_quat = self.normalize_quaternion(self.ground_truth_quats_current)
        gt_quat = gt_quat[:24]
        dot_product = np.sum(rendered_quat * gt_quat, axis=1)
        angle_per_joint = 2 * np.arccos(np.clip(dot_product, -1.0, 1.0))
        avg_angle = np.mean(angle_per_joint) # could be enhanced using joint weights
        return avg_angle
    
    def normalize_quaternion(self, q: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        return q / np.clip(norm, 1e-8, None)

class RewardCallback(BaseCallback):
    def __init__(self, num_envs, gesture_names, plot_freq=100, save_path='/home/dgusain/Bot_hand/figs/05_Gen/', verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.num_envs = num_envs
        self.gesture_names = gesture_names
        self.plot_freq = plot_freq  # Number of episodes between plot saves
        self.save_path = save_path
        self.gesture_env_rewards = {gesture: {env_idx: [] for env_idx in range(num_envs)} for gesture in self.gesture_names}
        self.gesture_env_avg = {gesture: {env_idx: 0.0 for env_idx in range(num_envs)} for gesture in self.gesture_names}

        self.episode_rewards = [[] for _ in range(num_envs)]
        self.episode_counts = [0 for _ in range(num_envs)]
        self.all_rewards = []  # List to store all episode rewards
        self.neg_threshold = -3

        # Ensure the save_path directory exists
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        rewards = self.locals.get('rewards', [])
        infos = self.locals.get('infos', [])
        episode_rewards = []

        for i, (done, reward, info) in enumerate(zip(dones, rewards, infos)):
            gesture = info.get('gesture', None)
            if gesture is not None:
                self.gesture_env_rewards[gesture][i].append(reward)
                self.gesture_env_avg[gesture][i] = np.mean(self.gesture_env_rewards[gesture][i])

                if done:
                    episodes = len(self.gesture_env_rewards[gesture][i])
                    self.episode_rewards[i].append(reward)
                    self.episode_counts[i] += 1
                    total_reward = np.sum(self.episode_rewards[i])
                    total_reward = max(total_reward, self.neg_threshold)
                    avg_reward = np.mean(self.episode_rewards[i])
                    max_reward = np.max(self.episode_rewards[i])
                    
                    logging.info(f"Episode {self.episode_counts[i]} | Env {i+1} | "
                             f"Avg. Reward: {avg_reward:.2f} | Max Reward: {max_reward:.2f}")
                    if self.episode_counts[i] % self.plot_freq == 0:
                        best_env_idx = max(self.gesture_env_avg[gesture], key=lambda x: self.gesture_env_avg[gesture][x])
                        best_rewards = self.gesture_env_rewards[gesture][best_env_idx]
                        self.plot_rewards(best_rewards, gesture,self.episode_counts[i])

        return True

    def plot_rewards(self, rewards, gesture,ep):
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, label=f"Total Reward per Episode for Gesture {gesture}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"Training Reward Progress up to Episode {ep} for Gesture '{gesture}'")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plot_filename = os.path.join(self.save_path, f"training_rewards_episode_{ep}_gesture_{gesture}.png")
        plt.savefig(plot_filename)
        plt.close()
        logging.info(f"Saved training plot to {plot_filename}")

    def plot_final_rewards(self):
        for gesture, env_rewards in self.gesture_env_rewards.items():
            best_env_idx = max(self.gesture_env_avg[gesture], key=lambda x: self.gesture_env_avg[gesture][x])
            rewards = env_rewards[best_env_idx]
            self.plot_rewards(rewards, gesture)

def make_env():
    def _init():
        env = HandEnv()
        return env
    return _init

def inference(gpu_id, gesture_key):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'     
    env = DummyVecEnv(make_env())
    env = VecMonitor(env)
    with open("quat_dict.pkl", 'rb') as file:
        grt = pickle.load(file)  # { "a": [[q1], [q2], ...], "c": [[q1], [q2], ...], ... }
        gesture_names = list(grt.keys())

    model_path = "/home/dgusain/Bot_hand/agents/agent_tf_05_Gen"
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"No saved model found at {model_path}.zip")

    model = RecurrentPPO.load(model_path, device=device)
    obs = env.reset(options={'gesture': gesture_key})
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    eval_rewards = []
    for step in range(10):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        # env.render()  # Consider disabling rendering for faster inference
        logging.info(f"Step: {step}, Action: {action}, Reward: {reward}, Done: {done}")
        eval_rewards.append(reward)
        if done.any():
            obs = env.reset(options={'gesture': gesture_key})
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
        else:
            episode_starts = np.zeros((1,), dtype=bool)

def train_on_gpu(gpu_id: int, num_envs: int, episodes: int, steps_per_episode: int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    env = VecMonitor(env)
    with open("quat_dict.pkl", 'rb') as file:
        grt = pickle.load(file)  # { "a": [[q1], [q2], ...], "c": [[q1], [q2], ...], ... }
    gesture_names = list(grt.keys())

    total_timesteps = episodes * num_envs  # Each episode is a single step

    model = RecurrentPPO(
        CustomACLstmPolicy,
        env,
        verbose=1,
        device=device,
        ent_coef=0.01,
        learning_rate=0.0003,
        clip_range=0.2,
        n_steps=steps_per_episode,  # Each episode is a single step
        batch_size= num_envs * steps_per_episode,  # batch_size should be a multiple of n_steps * num_envs
        gamma=0.998,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,
        use_sde=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  
        save_path="/home/dgusain/Bot_hand/agents/checkpoints/05_GenP/",  
        name_prefix="agent_tf"  
    )
    plot_frequency = 1000
    save_plots_path = "/home/dgusain/Bot_hand/figs/05_GenP/"  
    reward_callback = RewardCallback(
        num_envs=num_envs,
        gesture_names=gesture_names,
        plot_freq=plot_frequency,
        save_path=save_plots_path
    )
    callback = CallbackList([reward_callback, checkpoint_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10)
    model.save(f"/home/dgusain/Bot_hand/agents/agent_tf5_GenP")
    print("Model saved succesfully as agent_tf5_GenP")
    #reward_callback.plot_final_rewards() no need for this right now. 

    return callback

def main():
    gpu_id = 1
    num_envs = min(512, cpu_count())  # Use up to 8 or available CPU cores
    episodes = 10000  # Adjusted to compensate for single-step episodes
    steps_per_episode = 1  # Each episode is a single step

    train_on_gpu(gpu_id, num_envs, episodes, steps_per_episode)

if __name__ == "__main__":
    main()
