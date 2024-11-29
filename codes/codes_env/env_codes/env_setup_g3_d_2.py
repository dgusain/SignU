# Reset the env state back to initial, after each episode.
import gymnasium as gym
from gymnasium import spaces
from statistics import median
import numpy as np
import mujoco_py
import os
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
import logging
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Configure logging
logging.basicConfig(level=logging.INFO)

class HandEnv(gym.Env):
    def __init__(self):
        super(HandEnv, self).__init__()
        xml_path = os.path.expanduser('/home/easgrad/dgusain/Bot_hand/bot_hand.xml')
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
            3.0,  # ID 2: ForeFingerJoint3 (MCP)
            2.5,  # ID 3: ForeFingerJoint2 (MCP proximal)
            2.0,  # ID 4: ForeFingerJoint1 (PIP)
            1.0,  # ID 5: ForeFingerJoint0 (DIP)
            3.0,  # ID 6: MiddleFingerJoint3 (MCP)
            2.5,  # ID 7: MiddleFingerJoint2 (MCP proximal)
            2.0,  # ID 8: MiddleFingerJoint1 (PIP)
            1.0,  # ID 9: MiddleFingerJoint0 (DIP)
            3.0,  # ID 10: RingFingerJoint3 (MCP)
            2.5,  # ID 11: RingFingerJoint2 (MCP proximal)
            2.0,  # ID 12: RingFingerJoint1 (PIP)
            1.0,  # ID 13: RingFingerJoint0 (DIP)
            1.5,  # ID 14: LittleFingerJoint4 (CMC Joint rotation)
            3.0,  # ID 15: LittleFingerJoint3 (MCP)
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
        # Convertiing joint weights to a tensor
        self.joint_weights = torch.tensor(joint_weights, dtype=torch.float32)
        # Normalize the weights so that their sum equals 1
        self.joint_weights /= self.joint_weights.sum()

        self.initial_state = self.sim.get_state()
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        self.target_threshold = 100
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
        #self.sim.reset()
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        self.steps_taken = 0
        obs = torch.from_numpy(self.sim.data.qpos[:24].astype(np.float32))
        return obs, {}

    def step(self, action: np.ndarray):
        # Rescale action using tensor operations for efficiency
        action_tensor = torch.from_numpy(action).float()
        # adding some Gaussian noise to the action to avoid getting stuck in local optima
        #noise = torch.normal(mean=0, std=0.05, size=action_tensor.size())
        #action_tensor += noise
        rescaled_action = self.actuator_min + (action_tensor + 1) * (self.actuator_max - self.actuator_min) / 2
        # rescaled_action[0:2] = 0
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
            if confidence > 80:
                return (confidence-50)/2.0 # encouraging the model in the right direction
            else:
                return (confidence-50)/4.0 # lesser reward

    def calculate_confidence(self, state: torch.Tensor) -> torch.Tensor:
        rendered_quat = self.get_rendered_pose_quaternion()
        rendered_quat = self.normalize_quaternion(rendered_quat)
        gt_quat = self.normalize_quaternion(self.ground_truth_quats)
        similarity = torch.abs(torch.sum(rendered_quat * gt_quat, dim=1)) * 100
        weighted_similarity = similarity * self.joint_weights
        avg_confidence = weighted_similarity.sum()
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
        self.avg_rewards = {}  # Dictionary to store average rewards, key: block_num
        self.iteration_num = 0     # Counter for iterations
        self.rewards_list = []
        self.median_rewards = {}

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
                        block_num = self.episode_num // self.avg_interval
                        self.median_rewards[block_num] = median_reward
                        print(f"Iteration: {self.iteration_num} | Episodes: {block_num * self.avg_interval} | Median Reward: {median_reward:.2f} | Max Reward: {self.max_reward:.2f}")
                        # Reset sum and count
                        self.rewards_list = []
        return True

    def _on_training_end(self) -> None:
        """
        Called at the end of training. If there are remaining episodes, compute and store the average.
        """
        if len(self.rewards_list) > 0:
            median_reward = median(self.rewards_list)
            block_num = self.episode_num // self.avg_interval + 1
            self.median_rewards[block_num] = median_reward
            current_max = max(self.rewards_list)
            if current_max > self.max_reward:
                self.max_reward = current_max           
            print(f"Training End | Episodes: {self.episode_num} | Median Reward: {median_reward:.2f} | Max Reward: {self.max_reward:.2f}")
            self.rewards_list = []

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
        learning_rate=0.0005,
        clip_range=0.4,
        n_steps=num_steps,       # Steps per environment per update
        batch_size=4096,          # Increased batch size
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,
        use_sde=True,            # Use State Dependent Exploration for better exploration
    )

    callback = RewardCallback(avg_interval=100)

    try:
        # Train the model with the specified total timesteps
        model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10)
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise e

    # Save the final model
    model.save(f"/home/easgrad/dgusain/Bot_hand/agents/agent_g3d_{gpu_id}_2_weighted_try02")

    return callback  # Return the callback to access episode_rewards

def main():
    gpu_id = 4          # Set your GPU ID here
    num_envs = 4        # Number of parallel environments (adjust based on your GPU memory)
    n_iter = 500         # Number of iterations you want to run
    n_steps = 1024      # Steps per environment per update
    total_timesteps = n_iter * n_steps * num_envs  # Total training timesteps
    save_interval = 25  # Not used in this optimized version

    # Train the model and get the callback with rewards data
    callback = train_on_gpu(gpu_id, num_envs, total_timesteps, n_steps, save_interval)

    # Extract block numbers and average rewards from the callback
    block_numbers = list(callback.median_rewards.keys())
    block_rewards = list(callback.median_rewards.values())

    # Calculate the corresponding episode numbers (100, 200, 300, ...)
    episode_numbers = [block_num * 100 for block_num in block_numbers]

    # Plotting the average rewards vs episodes
    plt.figure(figsize=(10, 6))
    plt.plot(episode_numbers, block_rewards, marker='o', linestyle='-', color='b')
    plt.xlabel('Episode Number')
    plt.ylabel('Median Reward (per 100 episodes)')
    plt.title('Median Reward vs Episode Number')
    plt.grid(True)
    plt.savefig('/home/easgrad/dgusain/Bot_hand/figs/fig_g3d_2_weighted_try02.png')  # Save the figure
    plt.close()  # Close the figure to free memory
    print("Plot saved as fig_g3d_2_weighted_try02.png")

if __name__ == "__main__":
    main()

