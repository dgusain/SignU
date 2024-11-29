import mujoco_py
import os
import time

# Load the model
xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

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

def set_joint_positions(joint_positions, steps=200, delay=0.02):
    for i in range(steps):
        for joint, target_position in joint_positions.items():
            current_position = sim.data.ctrl[joint]
            new_position = current_position + (target_position - current_position) * (i / steps)
            sim.data.ctrl[joint] = new_position
        sim.step()
        viewer.render()
        time.sleep(delay)

# Initial positions for J
asl_j_start_positions = {
    **{actuator: 1.6 for finger in ["MiddleFinger", "RingFinger", "ForeFinger"] for actuator in finger_actuators[finger]},
    # Positioning the fingers - more human
    finger_actuators["ForeFinger"][3]: 0.136,  # ForeFingerJoint3_act (knuckle) [-0.436 0.436] 
    finger_actuators["MiddleFinger"][3]: 0.136,  # MiddleFingerJoint3_act (knuckle) [-0.436 0.436] 
    finger_actuators["RingFinger"][3]: 0.136,  # RingFingerJoint3_act (knuckle) [-0.436 0.436] 

    # Positioning the finger in the palm - more human
    finger_actuators["ForeFinger"][0]: 1.0,  # ForeFingerJoint0_act (tip) [0.0 1.6] .
    finger_actuators["MiddleFinger"][0]: 1.0,  # MiddleFingerJoint0_act (tip) [0.0 1.6]    
    finger_actuators["RingFinger"][0]: 1.0,  # RingFingerJoint0_act (tip) [0.0 1.6]  
     
    # Position of thumb as in a closed fist
    finger_actuators["Thumb"][0]: 0.0,  # ThumbJoint0_act (distal) [-1.571 0] (distal moves away from lf in negative, and closer in positive)
    finger_actuators["Thumb"][1]: 0.3,  # ThumbJoint1_act (middle) [-0.524 0.524] (the middle lifts upward in positive, and goes downward close to the finger in negative)
    finger_actuators["Thumb"][2]: 0.262,   # ThumbJoint2_act (hub) [-0.262 0.262]	( the hub tilts left towards lf on positive, and away from it on negative)
    finger_actuators["Thumb"][3]: 0.7,   # ThumbJoint3_act  (proximal) [0 1.309]	
    finger_actuators["Thumb"][4]: 0.8,   # ThumbJoint4_act (base) [-1.047 1.047] This one is responsible for moving the thumb forward. 
}

# Drawing the 'J' curve in the air
asl_j_curve_positions = [

     # Wrist 1: [-0.524 0.175], Wrist 0: [-0.785 0.611]
    {wrist_actuators["Wrist"][0]: -0.785, wrist_actuators["Wrist"][1]: -0.524},  # Starting position
    {wrist_actuators["Wrist"][0]: 0.611, wrist_actuators["Wrist"][1]: -0.524},  # Mid curve
    {wrist_actuators["Wrist"][0]: 0.611, wrist_actuators["Wrist"][1]: 0.175},  # End of curve
]

asl_revert = {
    **{actuator: 0.0 for w in ["Wrist"] for actuator in wrist_actuators[w]},
    **{actuator: 0.0 for w in ["ForeFinger"] for actuator in finger_actuators[w]},
}

set_joint_positions(asl_j_start_positions, 200, 0.02)
#set_joint_positions(asl_revert, 200, 0.02)
for position in asl_j_curve_positions:
    set_joint_positions(position, 100, 0.01)

# Keep the simulation running to view the result
while True:
    viewer.render()
    time.sleep(0.01)

