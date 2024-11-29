# calculates the reward function only when the episode terminates, which happens when the max steps has reached or the target pose has been reached. 
# Penalization reward = -1, for every extra step taken. 
# Idea: accumulate all penal rewards, and add to the final reward, hence pushing agent to use fewer steps. 
# implemented in v3. 
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco_py
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from multiprocessing import Process
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

class HandEnv(gym.Env):
    def __init__(self):
        super(HandEnv, self).__init__()
        xml_path = os.path.expanduser('/home/easgrad/dgusain/Bot_hand/bot_hand.xml')
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        self.target_threshold = 90
        self.max_steps = 100
        self.steps_taken = 0
        # Initialize min and max actuator ranges
        actuator_min = []
        actuator_max = []

        self.finger_actuators = {
            "ForeFinger": [self.sim.model.actuator_name2id(name) for name in [
                "ForeFingerJoint0_act", "ForeFingerJoint1_act", "ForeFingerJoint2_act", "ForeFingerJoint3_act"]],
            "MiddleFinger": [self.sim.model.actuator_name2id(name) for name in [
                "MiddleFingerJoint0_act", "MiddleFingerJoint1_act", "MiddleFingerJoint2_act", "MiddleFingerJoint3_act"]],
            "RingFinger": [self.sim.model.actuator_name2id(name) for name in [
                "RingFingerJoint0_act", "RingFingerJoint1_act", "RingFingerJoint2_act", "RingFingerJoint3_act"]],
            "LittleFinger": [self.sim.model.actuator_name2id(name) for name in [
                "LittleFingerJoint0_act", "LittleFingerJoint1_act", "LittleFingerJoint2_act", "LittleFingerJoint3_act","LittleFingerJoint4_act"]],
            "Thumb": [self.sim.model.actuator_name2id(name) for name in [
                "ThumbJoint0_act", "ThumbJoint1_act", "ThumbJoint2_act", "ThumbJoint3_act", "ThumbJoint4_act"]],
        }
        self.wrist_actuators = {
            "Wrist": [self.sim.model.actuator_name2id(name) for name in [
                "WristJoint0_act", "WristJoint1_act"]],
        }

        # Collect actuator min and max values for each actuator
        for finger, actuators in self.finger_actuators.items():
            for actuator_id in actuators:
                actuator_range = self.sim.model.actuator_ctrlrange[actuator_id]
                actuator_min.append(actuator_range[0])
                actuator_max.append(actuator_range[1])

        for w, wr_actuators in self.wrist_actuators.items():
            for actuator_id in wr_actuators:
                actuator_range = self.sim.model.actuator_ctrlrange[actuator_id]
                actuator_min.append(actuator_range[0])
                actuator_max.append(actuator_range[1])

        # Convert min and max values to numpy arrays for easier manipulation
        self.actuator_min = np.array(actuator_min)
        self.actuator_max = np.array(actuator_max)

        # Create normalized action space between [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.actuator_min),), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.steps_taken = 0
        return self.sim.data.qpos[:24].astype(np.float32), {}

    def step(self, action):
        # Rescale action from [-1, 1] to the actuator's actual range
        rescaled_action = self.actuator_min + (action + 1) * (self.actuator_max - self.actuator_min) / 2
        self.sim.data.ctrl[:] = rescaled_action
        self.sim.step()

        state = self.sim.data.qpos[:24].astype(np.float32)
        done = self.calculate_reward(state, False) or self.steps_taken >= self.max_steps
        if done:
            reward = self.calculate_reward(state,True)
        else:
            reward = -1
        
        self.steps_taken += 1
        truncated = False
        info = {}

        return state, reward, done, truncated, info

    def calculate_reward(self, state, flag):
        ground_truth_quat = self.get_ground_truth_quaternion()
        rendered_quat = self.get_rendered_pose_quaternion()
        confidence_scores = []
        for gt_q, r_q in zip(ground_truth_quat, rendered_quat):
            gt_q_norm = self.normalize_quaternion(gt_q)
            r_q_norm = self.normalize_quaternion(r_q)
            confidence_score = self.compute_quaternion_similarity(r_q_norm, gt_q_norm)
            confidence_scores.append(confidence_score)

        avg_confidence_score = np.mean(confidence_scores)
        # we may want to make this individual,if performance doesn't improve. 
        if flag: 
            return avg_confidence_score - 50  # Reward calculation
        else:
            return avg_confidence_score >= self.target_threshold

    
    def get_rendered_pose_quaternion(self):
        quaternions = []
        for joint in range(self.model.njnt):  # Iterate over all joints in the model
            quaternion = self.sim.data.qpos[self.model.jnt_qposadr[joint]:self.model.jnt_qposadr[joint] + 4]
            quaternions.append(quaternion)
        
        return np.array(quaternions)  # Return as a numpy array

    def compute_quaternion_similarity(self, q1, q2):
        dot_product = np.abs(np.dot(q1.flatten(), q2.flatten()))
        similarity = dot_product * 100
        return similarity

    def normalize_quaternion(self, q):
        """Normalize each quaternion to ensure it's a unit quaternion."""
        norm = np.linalg.norm(q)
        return q / norm if norm != 0 else q  # Prevent division by zero

    def get_ground_truth_quaternion(self):
        """
        Returns the ground truth quaternions for all joints as a numpy array.
        The provided quaternions will be used for comparison with the rendered pose.
        """
        ground_truth_quats = [
            [ 6.32658406e-04,  1.25473697e-03, -9.99718591e-03,  1.56702220e+00],
            [ 1.25473697e-03, -9.99718591e-03,  1.56702220e+00,  1.56730258e+00],
            [-0.00999719,  1.5670222,   1.56730258,  1.5674143 ],
            [ 1.5670222,   1.56730258,  1.5674143,  -0.00999801],
            [ 1.56730258,  1.5674143,  -0.00999801,  1.56701926],
            [ 1.5674143,  -0.00999801,  1.56701926,  1.56730194],
            [-0.00999801,  1.56701926,  1.56730194,  1.5674143 ],
            [ 1.56701926,  1.56730194,  1.5674143,  -0.00999756],
            [ 1.56730194,  1.5674143,  -0.00999756,  1.56701717],
            [ 1.5674143,  -0.00999756,  1.56701717,  1.56730177],
            [-0.00999756,  1.56701717,  1.56730177,  1.56741429],
            [1.56701717, 1.56730177, 1.56741429, 0.00868252],
            [ 1.56730177,  1.56741429,  0.00868252, -0.01000805],
            [ 1.56741429,  0.00868252, -0.01000805,  1.56708349],
            [ 0.00868252, -0.01000805,  1.56708349,  1.56730911],
            [-0.01000805,  1.56708349,  1.56730911,  1.56741508],
            [1.56708349, 1.56730911, 1.56741508, 0.49916711],
            [1.56730911, 1.56741508, 0.49916711, 0.50022545],
            [1.56741508, 0.49916711, 0.50022545, 0.25330455],
            [4.99167109e-01, 5.00225452e-01, 2.53304550e-01, 4.08440608e-04],
            [ 5.00225452e-01,  2.53304550e-01,  4.08440608e-04, -9.00000689e-01],
            [ 2.53304550e-01,  4.08440608e-04, -9.00000689e-01,  1.00000000e-01],
            [ 4.08440608e-04, -9.00000689e-01,  1.00000000e-01, -1.00000000e-01],
            [-0.90000069,  0.1,        -0.1,         0.01463168],
            [ 0.1,        -0.1,         0.01463168,  1.        ]
        ]

        return np.array(ground_truth_quats)  # Return as numpy array for comparison

from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):
    def __init__(self):
        super(RewardCallback, self).__init__()
        self.episode_num = 0

    def _on_step(self) -> bool:
        # Check if the episode is done
        if self.locals['dones'][0]:
            self.episode_num += 1
            reward = self.locals['rewards'][0]
            print(f"Episode: {self.episode_num} | Reward: {reward}")
        return True



def train_on_gpu(gpu_id, iterations_per_gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"GPU {gpu_id}: Using {device} device")

    env = HandEnv()
    model = RecurrentPPO('MlpLstmPolicy', env,verbose=1,device=device, ent_coef=0.01,  learning_rate=0.0001,  clip_range=0.3, n_steps=2056)
    callback = RewardCallback()

    for iteration in range(iterations_per_gpu):
        model.learn(total_timesteps=100, reset_num_timesteps=False,callback = callback,log_interval = None)

        print(f"GPU {gpu_id} - Iteration {iteration + 1}/{iterations_per_gpu} completed")
    
    # Save the model uniquely per GPU
    model.save(f"hand_pose_ppo_gpu_{gpu_id}")

def main():
    total_iterations = 1000  
    train_on_gpu(5, total_iterations)

if __name__ == "__main__":
    main()
