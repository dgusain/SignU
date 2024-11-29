import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco_py
import os

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

        self.finger_actuators = {
            "ForeFinger": [self.sim.model.actuator_name2id(name) for name in [
                "ForeFingerJoint0_act", "ForeFingerJoint1_act", "ForeFingerJoint2_act", "ForeFingerJoint3_act"]],
            "MiddleFinger": [self.sim.model.actuator_name2id(name) for name in [
                "MiddleFingerJoint0_act", "MiddleFingerJoint1_act", "MiddleFingerJoint2_act", "ForeFingerJoint3_act"]],
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
        info = {}
        truncated = False

        return state, reward, done, truncated, info

    def calculate_reward(self, state):
        ground_truth_quat = self.get_ground_truth_quaternion()
        rendered_quat = self.get_rendered_pose_quaternion()
        confidence_score = self.compute_quaternion_similarity(rendered_quat, ground_truth_quat)
        return confidence_score - 50

    def compute_quaternion_similarity(self, q1, q2):
        return np.dot(q1, q2)

    def get_ground_truth_quaternion(self):
        return np.array([1, 0, 0, 0])

    def get_rendered_pose_quaternion(self):
        return np.array([1, 0, 0, 0])
