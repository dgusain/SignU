import gymnasium as gym
from gymnasium import spaces
from statistics import median
import numpy as np
import mujoco_py
import os
import torch
from torch import nn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement
)
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import logging
import matplotlib.pyplot as plt  
import math
import wandb
import random

# Set up logging
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
    Simplified by removing self-attention and reducing residual blocks.
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        assert isinstance(observation_space, spaces.Box), "Observation space must be of type Box"
        super(CustomFE, self).__init__(observation_space, features_dim)

        self.initial_net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # Reduced number of residual blocks
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
    Custom Actor-Critic Policy with LSTM for RecurrentPPO.
    Integrates the CustomFeaturesExtractor.
    """
    def __init__(self, *args, **kwargs):
        super(CustomACLstmPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomFE,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256], vf=[256]),  # Simplified architecture
            activation_fn=nn.ReLU,
            lstm_hidden_size=128  # Standard hidden size
        )

class HandEnv(gym.Env):
    def __init__(self):
        super(HandEnv, self).__init__()
        xml_path = os.path.expanduser('/home/dgusain/Bot_hand/bot_hand.xml')
        logging.info(f"Attempting to load Mujoco model from: {xml_path}")

        if not os.path.isfile(xml_path):
            logging.error(f"XML file not found at {xml_path}.")
            raise FileNotFoundError(f"XML file not found at {xml_path}.")

        try:
            self.model = mujoco_py.load_model_from_path(xml_path)
            self.sim = mujoco_py.MjSim(self.model)
            logging.info("Mujoco model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Mujoco model: {e}")
            raise RuntimeError(f"Failed to load Mujoco model: {e}")

        joint_weights = [
            0.5,  # ID 0: WristJoint1
            0.5,  # ID 1: WristJoint0
            1.5,  # ID 2: ForeFingerJoint3 (MCP)
            2.5,  # ID 3: ForeFingerJoint2 (MCP proximal)
            2.0,  # ID 4: ForeFingerJoint1 (PIP)
            1.0,  # ID 5: ForeFingerJoint0 (DIP)
            1.5,  # ID 6: MiddleFingerJoint3 (MCP)
            2.5,  # ID 7: MiddleFingerJoint2 (MCP proximal)
            2.0,  # ID 8: MiddleFingerJoint1 (PIP)
            1.0,  # ID 9: MiddleFingerJoint0 (DIP)
            1.5,  # ID 10: RingFingerJoint3 (MCP)
            2.5,  # ID 11: RingFingerJoint2 (MCP proximal)
            2.0,  # ID 12: RingFingerJoint1 (PIP)
            1.0,  # ID 13: RingFingerJoint0 (DIP)
            1.5,  # ID 14: LittleFingerJoint4 (CMC Joint rotation)
            1.5,  # ID 15: LittleFingerJoint3 (MCP)
            2.5,  # ID 16: LittleFingerJoint2 (MCP proximal)
            2.0,  # ID 17: LittleFingerJoint1 (PIP)
            1.0,  # ID 18: LittleFingerJoint0 (DIP)
            1.5,  # ID 19: ThumbJoint4 (CMC Joint abduction/adduction)
            1.5,  # ID 20: ThumbJoint3 (CMC Joint flexion/extension)
            1.5,  # ID 21: ThumbJoint2 (CMC Joint rotation)
            3.0,  # ID 22: ThumbJoint1 (MCP)
            0.5,  # ID 23: ThumbJoint0 (IP)
            0.0   # ID 24: None
        ]
        # Converting joint weights to a tensor
        self.joint_weights = torch.tensor(joint_weights, dtype=torch.float32)
        # Normalize the weights so that their sum equals 1
        self.joint_weights /= self.joint_weights.sum()

        self.initial_state = self.sim.get_state()
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32)  # 24 positions + 24 velocities
        self.target_threshold = 99
        self.max_steps = 100
        self.steps_taken = 0
        self.sim_coeff = 0.5
        self.ang_coeff = 1 - self.sim_coeff

        # Initialize actuator ranges
        actuator_ranges = self.sim.model.actuator_ctrlrange
        self.actuator_min = torch.tensor(actuator_ranges[:, 0], dtype=torch.float32)
        self.actuator_max = torch.tensor(actuator_ranges[:, 1], dtype=torch.float32)

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.actuator_min.shape[0],), dtype=np.float32)
        self.ground_truth_quats = torch.tensor(self.get_ground_truth_quaternion(), dtype=torch.float32)

        # Initialize previous action for smoothness penalty
        self.prev_action = torch.zeros_like(self.actuator_min)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        self.steps_taken = 0
        qpos = self.sim.data.qpos[:24].astype(np.float32)
        qvel = self.sim.data.qvel[:24].astype(np.float32)
        obs = np.concatenate([qpos, qvel])
        self.prev_action = torch.zeros_like(self.actuator_min)
        return obs, {}

    def step(self, action: np.ndarray):
        action_tensor = torch.from_numpy(action).float()
        noise = torch.normal(mean=0.0, std=0.05, size=action_tensor.size())
        rescaled_action = self.actuator_min + (action_tensor + 1) * (self.actuator_max - self.actuator_min) / 2
        rescaled_action += noise
        rescaled_action = torch.clamp(rescaled_action, self.actuator_min, self.actuator_max)
        self.sim.data.ctrl[:] = rescaled_action.numpy()
        self.sim.step()

        # Fetch state as a numpy array including positions and velocities
        qpos = self.sim.data.qpos[:24].astype(np.float32)
        qvel = self.sim.data.qvel[:24].astype(np.float32)
        state = np.concatenate([qpos, qvel])

        # Calculate done condition
        done = self.calculate_done(torch.from_numpy(state[:24])) or self.steps_taken >= self.max_steps
        if done:
            reward = self.calculate_reward(torch.from_numpy(state[:24]), action_tensor, flag=True).item()
        else:
            reward = -1.0  # Penalize extra steps

        self.steps_taken += 1
        truncated = False
        info = {}

        return state, reward, done, truncated, info

    def calculate_done(self, state: torch.Tensor) -> bool:
        confidence = self.calculate_confidence(state)
        return confidence >= self.target_threshold

    def calculate_reward(self, state: torch.Tensor, action: torch.Tensor, flag: bool) -> torch.Tensor:
        confidence = self.calculate_confidence(state)
        
        # Base reward
        if flag:
            base_reward = confidence - 50
        else:
            base_reward = (confidence - 50) / 2.0 if confidence > 85 else (confidence - 50) / 4.0
        
        # Smoothness penalty
        action_diff = action - self.prev_action
        smoothness_penalty = torch.norm(action_diff, p=2)
        self.prev_action = action.clone()
        
        # Energy penalty
        energy_penalty = torch.norm(action, p=2)
        
        # Total reward with adjusted penalty coefficients
        total_reward = base_reward - 0.005 * smoothness_penalty - 0.005 * energy_penalty  # Reduced penalties
        
        return total_reward

    def calculate_confidence(self, state: torch.Tensor) -> torch.Tensor:
        # Retrieve and normalize quaternions
        rendered_quat = self.get_rendered_pose_quaternion()
        rendered_quat = self.normalize_quaternion(rendered_quat)
        gt_quat = self.normalize_quaternion(self.ground_truth_quats)
        
        # Compute dot product between rendered and ground truth quaternions for each joint
        dot_product = torch.sum(rendered_quat * gt_quat, dim=1)  # Shape: [24]
        
        # Similarity Calculation
        similarity_per_joint = torch.abs(dot_product)  # [24]
        weighted_similarity = similarity_per_joint * self.joint_weights  # [24]
        similarity_score = weighted_similarity.sum() * 100  # Scalar, 0-100
        
        # Angular Displacement Calculation
        angular_displacement = 2 * torch.acos(torch.clamp(dot_product, -1.0, 1.0))  # [24], radians
        scaled_displacement = (angular_displacement / (2 * math.pi)) * 100  # [24], 0-100
        scaled_displacement = torch.clamp(scaled_displacement, 0.0, 100.0)  # Ensure values are within [0, 100]
        weighted_angular_disp = scaled_displacement * self.joint_weights  # [24]
        angular_disp_score = weighted_angular_disp.sum()  # Scalar, 0-100
        
        # Invert Angular Displacement Score to Reflect Inverse Relationship
        inverted_angular_disp_score = 100 - angular_disp_score  # Scalar, 0-100
        
        # Combine Similarity and Inverted Angular Displacement
        avg_confidence = (similarity_score * self.sim_coeff) + (inverted_angular_disp_score * self.ang_coeff)  # 0-100
        
        return avg_confidence

    def get_rendered_pose_quaternion(self) -> torch.Tensor:
        quaternions = []
        for joint in range(self.model.njnt):
            q = self.sim.data.qpos[self.model.jnt_qposadr[joint]:self.model.jnt_qposadr[joint] + 4]
            quaternions.append(q)
        # Convert list of arrays to a single NumPy array
        quaternions_np = np.array(quaternions, dtype=np.float32)
        # Convert NumPy array to PyTorch tensor
        return torch.from_numpy(quaternions_np)

    def normalize_quaternion(self, q: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(q, dim=1, keepdim=True)
        return q / norm.clamp(min=1e-8)  # Prevent division by zero

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

class CustomWandbCallback(BaseCallback):
    def __init__(
        self,
        wandb_project: str,
        wandb_run_name: str,
        log_freq: int = 1000,
        save_freq: int = 10000,
        model_save_path: str = "./wandb_models/",
        verbose: int = 1,
        **kwargs
    ):
        super(CustomWandbCallback, self).__init__(verbose)
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.model_save_path = model_save_path
        self.kwargs = kwargs  # Store additional parameters
        os.makedirs(self.model_save_path, exist_ok=True)

    def _init_callback(self) -> None:
        wandb.init(
            project=self.wandb_project,
            name=self.wandb_run_name,
            config=self.model.get_parameters(),  # Pass the model parameters
            **self.kwargs  # Pass additional arguments
        )
        if self.verbose > 0:
            print(f"Wandb run '{self.wandb_run_name}' initialized.")

    def _on_step(self) -> bool:
        # Log custom metrics
        if self.n_calls % self.log_freq == 0:
            log_dict = {
                "step": self.num_timesteps,
                "learning_rate": self.model.learning_rate(self.model._current_progress_remaining),
                "entropy_loss": self.model.logger.name_to_value.get("train/entropy_loss", 0),
                "value_loss": self.model.logger.name_to_value.get("train/value_loss", 0),
                "policy_gradient_loss": self.model.logger.name_to_value.get("train/policy_gradient_loss", 0),
                "approx_kl": self.model.logger.name_to_value.get("train/approx_kl", 0),
                "clip_fraction": self.model.logger.name_to_value.get("train/clip_fraction", 0),
                "clip_range": self.model.clip_range,
                "explained_variance": self.model.logger.name_to_value.get("train/explained_variance", 0),
                "log_reward_per_model": self.model.logger.name_to_value.get("rollout/ep_rew_mean", 0),
                "log_eval_reward": self.model.logger.name_to_value.get("eval/mean_reward", 0)

            }
            wandb.log(log_dict, step=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        wandb.finish()
        if self.verbose > 0:
            print(f"Wandb run '{self.wandb_run_name}' finished.")

def make_env(seed: int = None):
    """Utility function to create a HandEnv with a fixed seed."""
    def _init():
        env = HandEnv()
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def train_on_gpu(gpu_id: int, num_envs: int, total_timesteps: int, num_steps: int, save_interval: int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"GPU {gpu_id}: Using {device} device")

    # Set seeds for reproducibility
    seed = 42 + gpu_id  # Different seed for each GPU
    env = SubprocVecEnv([make_env(seed=seed + i) for i in range(num_envs)])  
    env = VecMonitor(env)

    # Define a separate evaluation environment with the same seed
    eval_env = DummyVecEnv([make_env(seed=seed + num_envs)])  
    eval_env = VecMonitor(eval_env)

    # Set seeds for torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = RecurrentPPO(
        CustomACLstmPolicy,  # Use the custom policy
        env,
        verbose=1,
        device=device,
        ent_coef=0.01,              # Consider using a schedule if exploration is insufficient
        learning_rate=linear_schedule(3e-4),
        clip_range=0.2,             # Lower clipping range
        n_steps=2048,               # Increased steps per environment per update
        batch_size=4096,            # Adjusted to a multiple of num_envs
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,
        use_sde=True,               # Use State Dependent Exploration
        normalize_advantage=True,   # Normalize advantages
        tensorboard_log=f"/home/dgusain/Bot_hand/dactyl/logs/02_gpu{gpu_id}"  # Enable TensorBoard logging
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_interval * num_steps * num_envs, 
        save_path=f"/home/dgusain/Bot_hand/dactyl/checkpoints/02_checkpoints_gpu{gpu_id}", 
        name_prefix="agent_checkpoint"
    )
    
    # Define EvalCallback with StopTrainingOnNoModelImprovement
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=100,  # Increased patience
        min_evals=20,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"/home/dgusain/Bot_hand/dactyl/02_best_models_gpu{gpu_id}",
        log_path=f"/home/dgusain/Bot_hand/dactyl/eval_logs_gpu{gpu_id}",
        eval_freq=5000,             # Evaluate every 5000 steps
        deterministic=True,
        render=False
        #callback_after_eval=stop_callback  # Integrate stopping callback
    )
    
    # Initialize the Custom Wandb Callback with comprehensive logging
    custom_wandb_callback = CustomWandbCallback(
        wandb_project="dactyl_mimic",
        wandb_run_name=f"gpu{gpu_id}_run",
        log_freq=1000,              # Log metrics every 1000 steps
        save_freq=save_interval * num_steps * num_envs,  # Save frequency aligned with checkpoint_callback
        model_save_path=f"/home/dgusain/Bot_hand/dactyl/02_wandb_models_gpu{gpu_id}/",
        verbose=1
    )
    
    # Combine all callbacks into a CallbackList
    callback = CallbackList([
        checkpoint_callback, 
        eval_callback, 
        custom_wandb_callback
    ])

    try:
        # Initialize W&B
        wandb.init(
            project="dactyl_mimic",
            name=f"gpu{gpu_id}_run",
            config={
                "algorithm": "RecurrentPPO",
                "env": "HandEnv",
                "total_timesteps": total_timesteps,
                "learning_rate": 3e-4,
                "batch_size": 4096,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "max_grad_norm": 0.5,
                "vf_coef": 0.5,
                "use_sde": True,
                "normalize_advantage": True,
                "n_steps": 2048,
                "num_envs": num_envs,
                "seed": seed
            }
        )
        # Start the training process
        model.learn(
            total_timesteps=total_timesteps, 
            callback=callback, 
            log_interval=10
        )
    except Exception as e:
        logging.error(f"Error during training on GPU {gpu_id}: {e}")
        raise e
    finally:
        wandb.finish()

    # Save the final model
    model.save(f"/home/dgusain/Bot_hand/dactyl/agents/final/02_agent_gpu{gpu_id}")

    return None  # Removed returning the callback as VecMonitor handles reward tracking

def train_distributed(num_gpus: int, num_envs_per_gpu: int, total_timesteps: int, num_steps: int, save_interval: int):
    from multiprocessing import Process

    def worker(gpu_id, num_envs, total_timesteps, num_steps, save_interval):
        train_on_gpu(gpu_id, num_envs, total_timesteps, num_steps, save_interval)

    processes = []
    for gpu_id in range(num_gpus):
        p = Process(target=worker, args=(gpu_id, num_envs_per_gpu, total_timesteps, num_steps, save_interval))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def main():
    num_gpus = 1
    num_envs_per_gpu = 4  # Adjust based on GPU memory and environment complexity
    total_timesteps = 1_200_000  # Define total timesteps directly
    n_steps = 2048      # Steps per environment per update
    save_interval = 25  
    
    train_distributed(num_gpus, num_envs_per_gpu, total_timesteps, n_steps, save_interval)
    
    # Aggregated plotting can be handled separately if needed

if __name__ == "__main__":
    main()
