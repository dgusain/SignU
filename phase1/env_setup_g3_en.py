import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco_py
import os
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class HandEnv(gym.Env):
    def __init__(self):
        super(HandEnv, self).__init__()
        xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')
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

        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        self.target_threshold = 90
        self.max_steps = 100
        self.steps_taken = 0

        # Initialize actuator ranges
        actuator_ranges = self.sim.model.actuator_ctrlrange
        self.actuator_min = torch.tensor(actuator_ranges[:, 0], dtype=torch.float32)
        self.actuator_max = torch.tensor(actuator_ranges[:, 1], dtype=torch.float32)

        # Create normalized action space between [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.actuator_min.shape[0],), dtype=np.float32)

        # Precompute ground truth quaternions as a tensor
        self.ground_truth_quats = torch.tensor(self.get_ground_truth_quaternion(), dtype=torch.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.steps_taken = 0
        obs = torch.from_numpy(self.sim.data.qpos[:24].astype(np.float32))
        return obs, {}

    def step(self, action: np.ndarray):
        # Rescale action using tensor operations for efficiency
        action_tensor = torch.from_numpy(action).float()
        rescaled_action = self.actuator_min + (action_tensor + 1) * (self.actuator_max - self.actuator_min) / 2
        self.sim.data.ctrl[:] = rescaled_action.numpy()
        self.sim.step()

        # Fetch state as a tensor
        state = torch.from_numpy(self.sim.data.qpos[:24].astype(np.float32))
        
        # Calculate done condition
        done = self.calculate_done(state, flag=False) or self.steps_taken >= self.max_steps
        if done:
            reward = self.calculate_reward(state, flag=True).item()
        else:
            reward = -1.0  # Penalize extra steps

        self.steps_taken += 1
        truncated = False
        info = {}

        return state.numpy(), reward, done, truncated, info

    def calculate_done(self, state: torch.Tensor, flag: bool) -> bool:
        if flag:
            return False
        confidence = self.calculate_confidence(state)
        return confidence >= self.target_threshold

    def calculate_reward(self, state: torch.Tensor, flag: bool) -> torch.Tensor:
        if flag:
            confidence = self.calculate_confidence(state)
            return confidence - 50  # Reward calculation
        else:
            return torch.tensor(0.0)  # Intermediate steps have a fixed penalty

    def calculate_confidence(self, state: torch.Tensor) -> torch.Tensor:
        rendered_quat = self.get_rendered_pose_quaternion()
        rendered_quat = self.normalize_quaternion(rendered_quat)
        gt_quat = self.normalize_quaternion(self.ground_truth_quats)
        similarity = torch.abs(torch.sum(rendered_quat * gt_quat, dim=1)) * 100
        avg_confidence = similarity.mean()
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
    def __init__(self):
        super(RewardCallback, self).__init__()
        self.episode_num = 0
        self.max_reward = -np.inf

    def _on_step(self) -> bool:
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
                    print(f"Episode: {self.episode_num} | Reward: {reward:.2f} | Max Reward: {self.max_reward:.2f}")
        return True

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

def train_on_gpu(gpu_id: int, num_envs: int, total_timesteps: int, num_steps: int, save_interval: int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"GPU {gpu_id}: Using {device} device")

    # Choose between SubprocVecEnv and DummyVecEnv based on your needs
    # env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    env = DummyVecEnv([make_env() for _ in range(num_envs)])  # Use DummyVecEnv for testing
    env = VecMonitor(env)  # Use VecMonitor instead of individual Monitor wrappers

    # Initialize the RecurrentPPO model with optimized parameters
    model = RecurrentPPO(
        'MlpLstmPolicy',
        env,
        verbose=1,
        device=device,
        ent_coef=0.05,
        learning_rate=0.0001,
        clip_range=0.4,
        n_steps=num_steps,       # Steps per environment per update
        batch_size=256,          # Increased batch size
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,
        use_sde=True,            # Use State Dependent Exploration for better exploration
    )

    callback = RewardCallback()

    try:
        # Train the model with the specified total timesteps
        model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10)
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise e

    # Save the final model
    model.save(f"hand_pose_ppo_gpu_{gpu_id}")

def main():
    gpu_id = 0          # Set your GPU ID here
    num_envs = 2        # Number of parallel environments (adjust based on your GPU memory)
    n_iter = 10         # Number of iterations you want to run
    n_steps = 2048      # Steps per environment per update
    total_timesteps = n_iter * n_steps * num_envs  # Total training timesteps
    save_interval = 25  # Not used in this optimized version

    train_on_gpu(gpu_id, num_envs, total_timesteps, n_steps, save_interval)

if __name__ == "__main__":
    main()
