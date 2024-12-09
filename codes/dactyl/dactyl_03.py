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
    Increases network depth and incorporates layer normalization and attention.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        # Ensure the observation space is as expected
        assert isinstance(observation_space, spaces.Box), "Observation space must be of type Box"

        super(CustomFE, self).__init__(observation_space, features_dim)

        self.initial_net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # Add residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)  # Increased depth
        )

        # Add self-attention
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.final_net = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.initial_net(observations)
        x = self.residual_blocks(x)
        # Add a sequence dimension for attention
        x = x.unsqueeze(1)  # Shape: [batch_size, 1, 128]
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)    # Shape: [batch_size, 128]
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
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=nn.ReLU,
            lstm_hidden_size=128  # Size of LSTM hidden state
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
            if confidence > 85:
                base_reward = (confidence - 50) / 2.0
            else:
                base_reward = (confidence - 50) / 4.0
        
        # Smoothness penalty
        action_diff = action - self.prev_action
        smoothness_penalty = torch.norm(action_diff, p=2)
        self.prev_action = action.clone()
        
        # Energy penalty
        energy_penalty = torch.norm(action, p=2)
        
        # Total reward
        total_reward = base_reward - 0.01 * smoothness_penalty - 0.01 * energy_penalty
        
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

class RewardCallback(BaseCallback):
    """
    Custom callback for logging average rewards over every 100 episodes, episode numbers, and iteration numbers.
    Also stores average rewards for plotting.
    """

    def __init__(self, avg_interval=100):
        super(RewardCallback, self).__init__()
        self.episode_num = 0
        self.max_reward = -np.inf
        self.avg_interval = avg_interval  # Number of episodes per average
        self.sum_rewards = 0.0
        self.count_rewards = 0
        self.iteration_num = 0     
        self.rewards_list = []
        self.median_rewards = {}
        self.avg_rewards = {}  # New dictionary to store average rewards

    def _on_rollout_end(self) -> bool:
        """
        Called at the end of a rollout.
        Used to track the number of iterations.
        """
        self.iteration_num += 1
        return True

    def _on_step(self) -> bool:
        """
        Called at every step. Checks if any episode has finished and logs the reward.
        """
        # Retrieve 'dones' and 'rewards' from the current step
        dones = self.locals.get('dones', [])
        rewards = self.locals.get('rewards', [])

        # Check if any of the environments are done
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
                        # Reset sum and count
                        self.rewards_list = []
        return True

    def _on_training_end(self) -> None:
        """
        Called at the end of training. If there are remaining episodes, compute and store the average.
        """
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

class CustomWandbCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to Weights & Biases (W&B).
    """

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

        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_path, exist_ok=True)

    def _init_callback(self) -> None:
        """
        Called once before the training starts.
        Initializes the W&B run.
        """
        wandb.init(
            project=self.wandb_project,
            name=self.wandb_run_name,
            config=self.model.get_parameters(),
            **self.params  # Pass any additional parameters
        )
        if self.verbose > 0:
            print(f"Wandb run '{self.wandb_run_name}' initialized.")

    def _on_step(self) -> bool:
        """
        Called at every step.
        Logs metrics to W&B at specified intervals.
        """
        if self.n_calls % self.log_freq == 0:
            # Log custom metrics here
            # For example, you can log the current loss, entropy, etc.
            # These metrics need to be extracted from the model or environment
            # Here's a generic example:
            info = self.locals.get('infos', [{}])[0]
            reward = info.get('reward', 0)
            done = info.get('done', False)

            wandb.log({
                "step": self.num_timesteps,
                "reward": reward,
                # Add other metrics as needed
            })

        # Save the model at specified intervals
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.model_save_path, f"model_step_{self.num_timesteps}.zip")
            self.model.save(model_path)
            wandb.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.num_timesteps} to {model_path}")

        return True

    def _on_training_end(self) -> None:
        """
        Called at the end of training.
        Finalizes the W&B run.
        """
        wandb.finish()
        if self.verbose > 0:
            print(f"Wandb run '{self.wandb_run_name}' finished.")

def make_env():
    """Utility function to create a HandEnv without Monitor."""
    def _init():
        try:
            env = HandEnv()
            return env
        except Exception as e:
            logging.error(f"Failed to initialize HandEnv: {e}")
            raise e
    return _init

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def train_on_gpu(gpu_id: int, num_envs: int, total_timesteps: int, num_steps: int, save_interval: int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"GPU {gpu_id}: Using {device} device")
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])  
    env = VecMonitor(env)  # Use VecMonitor instead of individual Monitor wrappers

    # Define a separate evaluation environment
    eval_env = DummyVecEnv([make_env])  # Single environment for evaluation

    model = RecurrentPPO(
        CustomACLstmPolicy,  # Use the custom policy
        env,
        verbose=1,
        device=device,
        ent_coef=0.01,              # Reduced entropy coefficient
        learning_rate=linear_schedule(3e-4),
        clip_range=0.2,             # Lower clipping range
        n_steps=2048,               # Increased steps per environment per update
        batch_size=8192,            # Larger batch size
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,
        use_sde=True,               # Use State Dependent Exploration
        normalize_advantage=True,   # Normalize advantages
        tensorboard_log=f"/home/dgusain/Bot_hand/dactyl/logs/gpu{gpu_id}"  # Enable TensorBoard logging
    )

    reward_callback = RewardCallback(avg_interval=100)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_interval * num_steps * num_envs, 
        save_path=f"/home/dgusain/Bot_hand/dactyl/checkpoints/checkpoints_gpu{gpu_id}", 
        name_prefix="agent_checkpoint"
    )
    
    # Define EvalCallback with StopTrainingOnNoModelImprovement
    stop_callback = StopTrainingOnNoModelImprovement(
        monitor='mean_reward',
        min_delta=10,
        patience=50,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"/home/dgusain/Bot_hand/dactyl/best_models_gpu{gpu_id}",
        log_path=f"/home/dgusain/Bot_hand/dactyl/eval_logs_gpu{gpu_id}",
        eval_freq=5000,             # Evaluate every 5000 steps
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback  # Integrate stopping callback
    )
    
    # Initialize the Custom Wandb Callback
    custom_wandb_callback = CustomWandbCallback(
        wandb_project="dactyl_mimic",
        wandb_run_name=f"gpu{gpu_id}_run",
        log_freq=1000,              # Log metrics every 1000 steps
        save_freq=save_interval * num_steps * num_envs,  # Save frequency aligned with checkpoint_callback
        model_save_path=f"/home/dgusain/Bot_hand/dactyl/wandb_models_gpu{gpu_id}/",
        verbose=1
    )
    
    # Combine all callbacks into a CallbackList
    callback = CallbackList([
        reward_callback, 
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
                "batch_size": 8192,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "max_grad_norm": 0.5,
                "vf_coef": 0.5,
                "use_sde": True,
                "normalize_advantage": True,
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
    model.save(f"/home/dgusain/Bot_hand/dactyl/agents/final/agent_gpu{gpu_id}")

    return reward_callback  # Return the callback to access episode_rewards

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
    n_iter = 1500
    n_steps = 1024      # Steps per environment per update
    total_timesteps = n_iter * n_steps * num_envs_per_gpu * num_gpus  # Total training timesteps
    save_interval = 25  

    train_distributed(num_gpus, num_envs_per_gpu, total_timesteps, n_steps, save_interval)
    
    # Aggregated plotting can be handled separately if needed

if __name__ == "__main__":
    main()
