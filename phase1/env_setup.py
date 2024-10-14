import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco_py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

class HandEnv(gym.Env):
    def __init__(self):
        super(HandEnv, self).__init__()
        xml_path = os.path.expanduser('/home/easgrad/dgusain/Bot_hand/bot_hand.xml')
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)

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
        return self.sim.data.qpos[:24].astype(np.float32), {}

    def step(self, action):
        # Rescale action from [-1, 1] to the actuator's actual range
        rescaled_action = self.actuator_min + (action + 1) * (self.actuator_max - self.actuator_min) / 2

        self.sim.data.ctrl[:] = rescaled_action
        self.sim.step()

        state = self.sim.data.qpos[:24].astype(np.float32)
        reward = float(self.calculate_reward(state))
        done = reward >= 50
        truncated = False
        info = {}
        if done:
            self.logger.record("reward_per_episode", reward)

        return state, reward, done, truncated, info

    def calculate_reward(self, state):
        ground_truth_quat = self.get_ground_truth_quaternion()
        rendered_quat = self.get_rendered_pose_quaternion()

        # Normalize quaternions and compute per-joint confidence score
        confidence_scores = []
        for gt_q, r_q in zip(ground_truth_quat, rendered_quat):
            # Normalize individual quaternions
            gt_q_norm = self.normalize_quaternion(gt_q)
            r_q_norm = self.normalize_quaternion(r_q)
            confidence_score = self.compute_quaternion_similarity(r_q_norm, gt_q_norm)
            confidence_scores.append(confidence_score)

        # Average confidence scores across joints
        avg_confidence_score = np.mean(confidence_scores)
        
        return avg_confidence_score - 50  # Reward calculation

    def get_rendered_pose_quaternion(self):
        quaternions = []
        for joint in range(self.model.njnt):  # Iterate over all joints in the model
            quaternion = self.sim.data.qpos[self.model.jnt_qposadr[joint]:self.model.jnt_qposadr[joint] + 4]
            quaternions.append(quaternion)
        
        return np.array(quaternions)  # Return as a numpy array

    def compute_quaternion_similarity(self, q1, q2):
        """Compute the dot product for quaternion similarity."""
        return np.dot(q1.flatten(), q2.flatten())

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


# Monitoring and Logging Setup
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)
env = HandEnv()
model = PPO('MlpPolicy',env,verbose=1,device=device,ent_coef=0.01,learning_rate=0.0001,clip_range=0.3,n_steps=4096)

# Custom training loop with logging
n_iterations = 1000
iteration_size = 1024  # Number of timesteps per iteration
total_timesteps = n_iterations * iteration_size

for iteration in range(n_iterations):
    model.learn(total_timesteps=iteration_size, reset_num_timesteps=False)

    # After each iteration, evaluate the model and log the reward
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(1000):  # Specify the number of steps for evaluation
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            break
    print(f"Iteration {iteration + 1}/{n_iterations} - Reward: {reward}")

# Save the model
model.save("hand_pose_ppo")
