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
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import logging
import matplotlib.pyplot as plt  
import math

# New imports for Optuna and additional utilities
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
# Removed PyTorchLightningPruningCallback as it's not needed
from typing import Dict

# Set up logging
logging.basicConfig(
    filename='hyperparameter_tuning.log',  # Log file to track hyperparameters and results
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Ensure the 'figs_of_e' and 'models' directories exist
FIGS_DIR = '/home/easgrad/dgusain/Bot_hand/figs_of_e'
MODELS_DIR = '/home/easgrad/dgusain/Bot_hand/models'

os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

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
        out = self.fc2(x)
        out += residual  # Residual connection
        out = self.layer_norm(out)
        out = self.relu(out)
        return out

class CustomFE(BaseFeaturesExtractor):
    """
    Custom feature extractor for HandEnv.
    Increases network depth and incorporates layer normalization.
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
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],  # Corrected bracket
            activation_fn=nn.ReLU,
            lstm_hidden_size=128  # Size of LSTM hidden state
        )

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #self.sim.reset()
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        self.steps_taken = 0
        obs = torch.from_numpy(self.sim.data.qpos[:24].astype(np.float32))
        return obs, {}

    def step(self, action: np.ndarray):
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
        confidence = self.calculate_confidence(state)
        if flag:
            return confidence - 50  # Reward calculation
        else:
            if confidence > 85:
                return (confidence - 50) / 2.0  # Encouraging the model in the right direction
            else:
                return (confidence - 50) / 4.0  # Lesser reward

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
        self.avg_interval = avg_interval  # Number of episodes per average - not hyperparameter
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

def train_model(gpu_id: int, num_envs: int, total_timesteps: int, num_steps: int, hyperparams: Dict, trial: Trial):
    """
    Train the RecurrentPPO model with given hyperparameters.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"GPU {gpu_id}: Using {device} device")
    env = DummyVecEnv([make_env() for _ in range(num_envs)])  
    env = VecMonitor(env)  # Use VecMonitor instead of individual Monitor wrappers

    model = RecurrentPPO(
        CustomACLstmPolicy,  # Use the custom policy
        env,
        verbose=0,  # Set to 0 to reduce output during Optuna trials
        device=device,
        ent_coef=hyperparams['ent_coef'],
        learning_rate=hyperparams['learning_rate'],
        clip_range=hyperparams['clip_range'],
        n_steps=num_steps,          # Steps per environment per update
        batch_size=hyperparams['batch_size'],
        gamma=hyperparams['gamma'],
        gae_lambda=hyperparams['gae_lambda'],
        max_grad_norm=0.5,
        vf_coef=0.5,
        use_sde=True,               # Use State Dependent Exploration for better exploration
        # Removed lstm_hidden_size=128
    )

    callback = RewardCallback(avg_interval=100)

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10)
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise e

    # After training, check if max_reward reached
    max_reward = callback.max_reward
    if max_reward >= 50:
        model_filename = f"agent_trial_{trial.number}_reward_{max_reward:.2f}.zip"
        model_path = os.path.join(MODELS_DIR, model_filename)
        model.save(model_path)
        print(f"Model saved as {model_filename}")
        logging.info(f"Model saved: {model_filename} with max_reward: {max_reward:.2f}")

    return max_reward

def objective(trial: Trial):
    """
    Objective function for Optuna to optimize.
    Samples hyperparameters, trains the model, and returns the max reward achieved.
    """
    # Define hyperparameter search space
    hyperparams = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-4, 1e-1),
        'gamma': trial.suggest_uniform('gamma', 0.9, 0.999),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'batch_size': trial.suggest_categorical('batch_size', [2048, 4096, 8192])
    }

    # Log the hyperparameters
    logger.info(f"Trial {trial.number}: Hyperparameters: {hyperparams}")

    gpu_id = 6          
    num_envs = 4        
    n_iter = 500         
    n_steps = 1024      # Steps per environment per update
    total_timesteps = n_iter * n_steps * num_envs  # Total training timesteps

    # Train the model with the sampled hyperparameters
    max_reward = train_model(gpu_id, num_envs, total_timesteps, n_steps, hyperparams, trial)

    # Report the max_reward to Optuna
    trial.report(max_reward, step=trial.number)

    # If the trial is not complete yet, and if a pruner is set, decide to prune
    if trial.should_prune():
        logger.info(f"Trial {trial.number} pruned at step {trial.number}")
        raise optuna.exceptions.TrialPruned()

    # Log the result
    logger.info(f"Trial {trial.number}: Max Reward: {max_reward}")

    return max_reward

def main():
    # Create an Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),  # Use TPE sampler with a fixed seed for reproducibility
        study_name='RecurrentPPO_HandEnv_Study',
        storage=None,  # Use in-memory storage; for persistence, specify a database URL
        load_if_exists=False
    )

    # Optimize the objective function
    study.optimize(objective, n_trials=50)  # Set n_trials as needed

    # After optimization, print the best trial
    best_trial = study.best_trial

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save Optuna study figures
    try:
        import plotly.io as pio
        pio.renderers.default = 'png'

        pareto_front = optuna.visualization.plot_pareto_front(study, target_names=["Max Reward"])
        pareto_path = os.path.join(FIGS_DIR, 'pareto_front.png')
        pareto_front.write_image(pareto_path)
        print(f"Pareto front saved as {pareto_path}")
        logger.info(f"Pareto front saved as {pareto_path}")

        # Save other Optuna visualizations
        importance = optuna.visualization.plot_param_importances(study)
        importance_path = os.path.join(FIGS_DIR, 'param_importances.png')
        importance.write_image(importance_path)
        print(f"Parameter importances saved as {importance_path}")
        logger.info(f"Parameter importances saved as {importance_path}")

        # Save the optimization history
        history = optuna.visualization.plot_optimization_history(study)
        history_path = os.path.join(FIGS_DIR, 'optimization_history.png')
        history.write_image(history_path)
        print(f"Optimization history saved as {history_path}")
        logger.info(f"Optimization history saved as {history_path}")

        # Optionally, save all trials' data to a CSV file for further analysis
        trials_dataframe = study.trials_dataframe()
        csv_path = os.path.join(FIGS_DIR, 'trials_dataframe.csv')
        trials_dataframe.to_csv(csv_path, index=False)
        print(f"Trials dataframe saved as {csv_path}")
        logger.info(f"Trials dataframe saved as {csv_path}")

    except Exception as e:
        print(f"An error occurred while saving Optuna figures: {e}")
        logger.error(f"Error saving Optuna figures: {e}")

if __name__ == "__main__":
    main()
