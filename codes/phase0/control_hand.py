import mujoco_py
import os
import time

# Load the model
xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Actuator indices based on the uploaded actuators_for_joints.xml
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

# Function to close a finger
def close_finger(finger_actuator_ids):
    for i in range(100):  # Adjust the range for smoother movement
        for actuator_id in finger_actuator_ids:
            sim.data.ctrl[actuator_id] = i * 0.01  # Increment control value
        sim.step()
        viewer.render()
        time.sleep(0.01)

# Function to open a finger
def open_finger(finger_actuator_ids):
    for i in range(100):  # Adjust the range for smoother movement
        for actuator_id in finger_actuator_ids:
            sim.data.ctrl[actuator_id] = (100 - i) * 0.01  # Decrement control value
        sim.step()
        viewer.render()
        time.sleep(0.01)

# Main loop to close and open each finger sequentially
fingers = ["Thumb", "ForeFinger", "MiddleFinger", "RingFinger", "LittleFinger"]
for finger in fingers:
    close_finger(finger_actuators[finger])
    time.sleep(1)  # Pause for a moment before closing the next finger
    open_finger(finger_actuators[finger])
    time.sleep(1)  # Pause for a moment before opening the next finger
    
    
