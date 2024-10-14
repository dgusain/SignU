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

# Function to set joint positions
def set_joint_positions(joint_positions, steps=200, delay=0.02):
    for i in range(steps):
        for joint, target_position in joint_positions.items():
            current_position = sim.data.ctrl[joint]
            new_position = current_position + (target_position - current_position) * (i / steps)
            sim.data.ctrl[joint] = new_position
        sim.step()
        viewer.render()
        time.sleep(delay)

# Joint positions for the letter 'A' in ASL
asl_a_positions = {
    # Close all fingers
    **{actuator: 1.6 for finger in ["ForeFinger", "MiddleFinger", "RingFinger", "LittleFinger"] for actuator in finger_actuators[finger]},
    finger_actuators["ForeFinger"][3]: -0.01,
    finger_actuators["MiddleFinger"][3]: -0.01,
    finger_actuators["RingFinger"][3]: -0.01,
    finger_actuators["LittleFinger"][3]: -0.01,
    # Position the thumb
    finger_actuators["Thumb"][0]: -0.9,  # ThumbJoint0_act
    finger_actuators["Thumb"][1]: 0.0,   # ThumbJoint1_act
    finger_actuators["Thumb"][2]: 0.262,   # ThumbJoint2_act # fixed
    finger_actuators["Thumb"][3]: 0.5,   # ThumbJoint3_act # fixed
    finger_actuators["Thumb"][4]: 0.5,   # ThumbJoint4_act # fixed
}

# Joint positions for the letter 'B' in ASL
asl_b_positions = {
    # Extend all fingers
    **{actuator: 0.0 for finger in ["LittleFinger","RingFinger", "Thumb","MiddleFinger", "ForeFinger"] for actuator in finger_actuators[finger]},
    # Position the thumb across the palm
    finger_actuators["Thumb"][0]: 0.0,  # ThumbJoint0_act (distal) [-1.571 0] (distal moves away from lf in negative, and closer in positive)
    finger_actuators["Thumb"][1]: -0.5,  # ThumbJoint1_act (middle) [-0.524 0.524] (the middle lifts upward in positive, and goes downward close to the finger in negative)
    finger_actuators["Thumb"][2]: 0.262,   # ThumbJoint2_act (hub) [-0.262 0.262]	( the hub tilts left towards lf on positive, and away from it on negative)
    finger_actuators["Thumb"][3]: 1.309,   # ThumbJoint3_act  (proximal) [0 1.309]	
    finger_actuators["Thumb"][4]: 1.047,   # ThumbJoint4_act (base) [-1.047 1.047] This one is responsible for moving the thumb forward. 
}

asl_revert = { **{actuator: 0.0 for finger in ["Thumb"] for actuator in finger_actuators[finger]}}

for i in range(5): 
	# Set the hand to display the letter 'A'
	set_joint_positions(asl_a_positions)
	time.sleep(0.02)  # Pause to view the letter 'A'
	
	#set_joint_positions(asl_revert,100, 0.01)
	# Set the hand to display the letter 'B'
	set_joint_positions(asl_b_positions)
	time.sleep(0.02)  # Pause to view the letter 'B'
	
	set_joint_positions(asl_revert,200, 0.01)


# Keep the simulation running to view the result
while True:
    viewer.render()
    time.sleep(0.01)

