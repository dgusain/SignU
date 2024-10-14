import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco_py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import json

class HandEnv(gym.Env):
    def __init__(self):
        super(HandEnv, self).__init__()
        xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)

        # Initialize min and max actuator ranges
        actuator_min = []
        actuator_max = []

        # Joint mapping between SMPL-X joints and MuJoCo model joints, including wrist
        self.joint_mapping = [
            ("ForeFingerJoint2", "F1"),  # Proximal
            ("ForeFingerJoint1", "F2"),  # Middle
            ("ForeFingerJoint0", "F3"),  # Distal
            ("MiddleFingerJoint2", "M1"),  # Proximal
            ("MiddleFingerJoint1", "M2"),  # Middle
            ("MiddleFingerJoint0", "M3"),  # Distal
            ("RingFingerJoint2", "R1"),  # Proximal
            ("RingFingerJoint1", "R2"),  # Middle
            ("RingFingerJoint0", "R3"),  # Distal
            ("LittleFingerJoint2", "L1"),  # Proximal
            ("LittleFingerJoint1", "L2"),  # Middle
            ("LittleFingerJoint0", "L3"),  # Distal
            ("ThumbJoint4", "T1"),  # Knuckle
            ("ThumbJoint2", "T2"),  # Middle
            ("ThumbJoint1", "T3")   # Distal
        ]

        self.finger_actuators = {
            "ForeFinger": [self.sim.model.actuator_name2id(name) for name in [
                "ForeFingerJoint0_act", "ForeFingerJoint1_act", "ForeFingerJoint2_act", "ForeFingerJoint3_act"]],
            "MiddleFinger": [self.sim.model.actuator_name2id(name) for name in [
                "MiddleFingerJoint0_act", "MiddleFingerJoint1_act", "MiddleFingerJoint2_act", "MiddleFingerJoint3_act"]],
            "RingFinger": [self.sim.model.actuator_name2id(name) for name in [
                "RingFingerJoint0_act", "RingFingerJoint1_act", "RingFingerJoint2_act", "RingFingerJoint3_act"]],
            "LittleFinger": [self.sim.model.actuator_name2id(name) for name in [
                "LittleFingerJoint0_act", "LittleFingerJoint1_act", "LittleFingerJoint2_act", "LittleFingerJoint3_act", "LittleFingerJoint4_act"]],
            "Thumb": [self.sim.model.actuator_name2id(name) for name in [
                "ThumbJoint0_act", "ThumbJoint1_act", "ThumbJoint2_act", "ThumbJoint3_act", "ThumbJoint4_act"]],
        }
        self.wrist_actuators = {
            "Wrist": [self.sim.model.actuator_name2id(name) for name in [
                "WristJoint0_act", "WristJoint1_act"]],
        }

        # Collect actuator min and max values for each actuator
        for actuators in (self.finger_actuators.values(), self.wrist_actuators.values()):
            for actuator_ids in actuators:
                for actuator_id in actuator_ids:
                    actuator_range = self.sim.model.actuator_ctrlrange[actuator_id]
                    actuator_min.append(actuator_range[0])
                    actuator_max.append(actuator_range[1])

        # Convert min and max values to numpy arrays for easier manipulation
        self.actuator_min = np.array(actuator_min)
        self.actuator_max = np.array(actuator_max)

        # Create normalized action space between [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.actuator_min),), dtype=np.float32)

        # Configure logging to use TensorBoard
        log_dir = "./logs/"
        os.makedirs(log_dir, exist_ok=True)
        self.logger = configure(log_dir, ["tensorboard"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        # Optionally, initialize the simulation to the ground truth pose
        # This depends on whether you want the agent to start from the ground truth
        # self.sim.data.qpos[:24] = self.get_ground_truth_quaternion().flatten()
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
            gt_q_norm = self.normalize_quaternion(gt_q)
            r_q_norm = self.normalize_quaternion(r_q)
            confidence_score = self.compute_quaternion_similarity(r_q_norm, gt_q_norm)
            confidence_scores.append(confidence_score)

        # Average confidence scores across joints
        avg_confidence_score = np.mean(confidence_scores)
        return avg_confidence_score - 50  # Reward calculation

    def get_rendered_pose_quaternion(self):
        """Get quaternions from the MuJoCo simulation."""
        quaternions = []
        for joint in range(self.model.njnt):  # Iterate over all joints in the model
            # Assuming each joint's quaternion is stored sequentially in qpos
            start = self.model.jnt_qposadr[joint]
            quaternion = self.sim.data.qpos[start:start + 4]
            quaternions.append(quaternion)
        return np.array(quaternions)

    def get_ground_truth_quaternion(self):
        """Return ground truth quaternions for all 24 joints, adding wrist joints."""
        json_file_path = '/home/ducky/Downloads/ASL_Fei/json_label_A/031.json'
        with open(json_file_path, 'r') as f:
            smplx_data = json.load(f)

        right_hand_pose = np.array(smplx_data['right_hand_pose']).reshape(-1, 3)

        # Initialize joint quaternions for all MuJoCo joints, setting default identity quaternions
        joint_quaternions = {self.sim.model.joint_id2name(i): np.array([1, 0, 0, 0]) for i in range(self.model.njnt)}

        # Fill in the quaternions using the joint mapping for SMPL-X joints
        for (mujoco_joint, smplx_joint) in self.joint_mapping:
            index = int(smplx_joint[1:]) - 1  # Extract the number from SMPL-X mapping, e.g., "F1" -> 0
            rotation_vector = right_hand_pose[index]
            quaternion = self.rotation_vector_to_quaternion(rotation_vector)
            joint_quaternions[mujoco_joint] = quaternion

        # For wrist joints, ensure they have identity quaternions
        for wrist_joint in ["WristJoint0", "WristJoint1"]:
            joint_quaternions[wrist_joint] = np.array([1, 0, 0, 0])  # Identity quaternion for wrist

        # Return quaternions in the correct order for MuJoCo model joints
        quaternions = []
        for joint_id in range(self.model.njnt):
            joint_name = self.sim.model.joint_id2name(joint_id)
            quaternions.append(joint_quaternions[joint_name])

        return np.array(quaternions)

    def rotation_vector_to_quaternion(self, rotation_vector):
        """Convert a rotation vector to a quaternion."""
        angle = np.linalg.norm(rotation_vector)
        if angle < 1e-6:  # small angle approximation
            return np.array([1, 0, 0, 0])  # Identity quaternion
        axis = rotation_vector / angle
        half_angle = angle / 2
        w = np.cos(half_angle)
        x, y, z = axis * np.sin(half_angle)
        return np.array([w, x, y, z])

    def compute_quaternion_similarity(self, q1, q2):
        """Compute the dot product for quaternion similarity."""
        # Ensure quaternions are numpy arrays
        q1 = np.array(q1)
        q2 = np.array(q2)
        dot_product = np.abs(np.dot(q1.flatten(), q2.flatten()))
        similarity = dot_product * 100
        return similarity

    def normalize_quaternion(self, q):
        """Normalize each quaternion to ensure it's a unit quaternion."""
        norm = np.linalg.norm(q)
        return q / norm if norm != 0 else q  # Prevent division by zero

# Instantiate the environment
env = HandEnv()

# Get the initial rendered quaternion and ground truth quaternion
rendered_quat = env.get_rendered_pose_quaternion()  # Call method from env instance
ground_truth_quat = env.get_ground_truth_quaternion()  # Call method from env instance

print("Initial Rendered Quaternion:")
print(rendered_quat)
print(len(rendered_quat))

print("\nGround Truth Quaternion:")
print(ground_truth_quat)
print(len(ground_truth_quat))

# Compute similarity for initial quaternions
for q1, q2 in zip(rendered_quat, ground_truth_quat):
    print(env.compute_quaternion_similarity(q1, q2))

# Define r_quat as a NumPy array to prevent AttributeError
r_quat = np.array([
    [6.32658406e-04,  1.25473697e-03, -9.99718591e-03,  1.56702220e+00],
    [1.25473697e-03, -9.99718591e-03,  1.56702220e+00,  1.56730258e+00],
    [-0.00999719,  1.5670222,   1.56730258,  1.5674143 ],
    [1.5670222,   1.56730258,  1.5674143,  -0.00999801],
    [1.56730258,  1.5674143,  -0.00999801,  1.56701926],
    [1.5674143,  -0.00999801,  1.56701926,  1.56730194],
    [-0.00999801,  1.56701926,  1.56730194,  1.5674143 ],
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
    [0.1,        -0.1,         0.01463168,  1.        ]
])

# Compute similarity for r_quat and ground truth
for q1, q2 in zip(r_quat, ground_truth_quat):
    print(env.compute_quaternion_similarity(q1, q2))

# The following parts are commented out as per the original code
# # Configure logging to use Tensorboard
# logger = configure(log_dir, ["tensorboard"])

# # Define the PPO model
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

# # Train the model and log progress to Tensorboard
# model.learn(total_timesteps=100000)

# # Save the model
# model.save("hand_pose_ppo")
