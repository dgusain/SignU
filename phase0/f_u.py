import mujoco_py
import os
import time

# Load the model
xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Actuator indices: joints and tendons
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

tendon_actuators = {
     "ForeFinger": [sim.model.actuator_name2id(name) for name in [
        "FFJ3r_motor", "FFJ3l_motor", "FFJ2u_motor", "FFJ2d_motor", "FFJ1u_motor", "FFJ1d_motor"]],
    "MiddleFinger": [sim.model.actuator_name2id(name) for name in [
        "MFJ3r_motor", "MFJ3l_motor", "MFJ2u_motor", "MFJ2d_motor", "MFJ1u_motor", "MFJ1d_motor"]],
    "RingFinger": [sim.model.actuator_name2id(name) for name in [
        "RFJ3r_motor", "RFJ3l_motor", "RFJ2u_motor", "RFJ2d_motor", "RFJ1u_motor", "RFJ1d_motor"]],
    "LittleFinger": [sim.model.actuator_name2id(name) for name in [
        "LFJ4u_motor", "LFJ4d_motor","LFJ3r_motor", "LFJ3l_motor", "LFJ2u_motor", "LFJ2d_motor", "LFJ1u_motor", "LFJ1d_motor"]],
    "Thumb": [sim.model.actuator_name2id(name) for name in [
        "THJ4a_motor", "THJ4c_motor","THJ3u_motor", "THJ3d_motor", "THJ2u_motor", "THJ2d_motor", "THJ1r_motor", "THJ1l_motor","THJ0r_motor", "THJ0l_motor"]],
}


# Function to close a finger
def close_finger(part_actuator_ids):
    for i in range(160):  # Adjust the range for smoother movement
        for actuator_id in part_actuator_ids:
            sim.data.ctrl[actuator_id] = i * 0.01  # Increment control value
        sim.step()
        viewer.render()
        time.sleep(0.01)

# Function to open a finger
def open_finger(part_actuator_ids):
    for i in range(160):  # Adjust the range for smoother movement
        for actuator_id in part_actuator_ids:
            sim.data.ctrl[actuator_id] = (100 - i) * 0.01  # Decrement control value
        sim.step()
        viewer.render()
        time.sleep(0.01)

def move_tendon(part_actuator_ids):
    for i in range(100):  # Adjust the range for smoother movement
        for actuator_id in part_actuator_ids:
            sim.data.ctrl[actuator_id] = (100 - i) * 0.01  # Increment control value
            print(actuator_id)
        sim.step()
        viewer.render()
        time.sleep(0.01)       
 

tendons = ["ForeFinger"]
for ten in tendons:
    move_tendon(tendon_actuators[ten])
# Main loop to close and open each finger sequentially
'''
fingers = ["ForeFinger","RingFinger", "LittleFinger"]
for finger in fingers:
    close_finger(finger_actuators[finger])
    #time.sleep(1)  # Pause for a moment before closing the next finger
    #open_finger(finger_actuators[finger])
    #time.sleep(1)  # Pause for a moment before opening the next finger
'''
while True:
    sim.step(0.1)
    viewer.render()
