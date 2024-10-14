import mujoco_py
import os
import time
from gymnasium import spaces
import numpy as np

# Load the model
xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
joint_positions = sim.data.qpos[:24]
#viewer = mujoco_py.MjViewer(sim)

# Actuator indices for the finger and thumb joints based on the uploaded actuators_for_joints.xml
finger_actuators = {
    "ForeFinger": [sim.model.actuator_name2id(name) for name in [
        "ForeFingerJoint0_act", "ForeFingerJoint1_act", "ForeFingerJoint2_act", "ForeFingerJoint3_act"]],
    "MiddleFinger": [sim.model.actuator_name2id(name) for name in [
        "MiddleFingerJoint0_act", "MiddleFingerJoint1_act", "MiddleFingerJoint2_act", "MiddleFingerJoint3_act"]],
    "RingFinger": [sim.model.actuator_name2id(name) for name in [
        "RingFingerJoint0_act", "RingFingerJoint1_act", "RingFingerJoint2_act", "RingFingerJoint3_act"]],
    "LittleFinger": [sim.model.actuator_name2id(name) for name in [
        "LittleFingerJoint0_act", "LittleFingerJoint1_act", "LittleFingerJoint2_act", "LittleFingerJoint3_act"]],
    "Thumb": [sim.model.actuator_name2id(name) for name in [
        "ThumbJoint0_act", "ThumbJoint1_act", "ThumbJoint2_act", "ThumbJoint3_act", "ThumbJoint4_act"]],
}

wrist_actuators = {
	"Wrist": [sim.model.actuator_name2id(name) for name in [
        "WristJoint0_act", "WristJoint1_act"]],
}

actuator_min = []
actuator_max = []

for finger, actuators in finger_actuators.items():
    print(f"Finger: {finger}, Actuator IDs: {actuators}")
    for actuator_id in actuators:
        actuator_range = sim.model.actuator_ctrlrange[actuator_id]  # Get min and max range for the actuator
        print(f"Actuator {actuator_id} range: {actuator_range}")
        actuator_min.append(actuator_range[0])
        actuator_max.append(actuator_range[1])

for w,wr_actuators in wrist_actuators.items():
    print(f"{w},  Actuator IDs: {wr_actuators}")
    for actuator_id in wr_actuators:
        actuator_range = sim.model.actuator_ctrlrange[actuator_id]  # Get min and max range for the actuator
        print(f"Actuator {actuator_id} range: {actuator_range}")
        actuator_min.append(actuator_range[0])
        actuator_max.append(actuator_range[1])
    
# Create action space with these ranges
action_space = spaces.Box(low=np.array(actuator_min), high=np.array(actuator_max), dtype=np.float32)

print("State space: ", joint_positions)
print("Action space: ",action_space)

