import mujoco_py
import os
import time
import numpy as np

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

# Initialize data storage
expert_states = []
expert_actions = []

# Function to set joint positions and record state-action pairs
def set_joint_positions(joint_positions, steps=200, delay=0.02):
    for i in range(steps):
        action = np.zeros(len(sim.model.actuator_ctrlrange))  # Initialize action array
        for finger, actuators in finger_actuators.items():
            for actuator in actuators:
                target_position = joint_positions.get(actuator, sim.data.ctrl[actuator])
                current_position = sim.data.ctrl[actuator]
                # Linear interpolation for smooth transition
                new_position = current_position + (target_position - current_position) * (i / steps)
                sim.data.ctrl[actuator] = new_position
                action[actuator] = new_position  # Record the action for this joint

        sim.step()
        viewer.render()
        time.sleep(delay)

        # Record the current state and action
        state = sim.data.qpos[:24].copy()  # Adjust based on your observation space
        expert_states.append(state)
        expert_actions.append(action.copy())

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
    finger_actuators["Thumb"][2]: 0.262, # ThumbJoint2_act
    finger_actuators["Thumb"][3]: 0.5,   # ThumbJoint3_act
    finger_actuators["Thumb"][4]: 0.5,   # ThumbJoint4_act
}

asl_revert = {**{actuator: 0.0 for finger in ["Thumb"] for actuator in finger_actuators[finger]}}

# Perform multiple cycles of forming and reverting the letter 'A'
for i in range(5):
    # Set the hand to display the letter 'A'
    set_joint_positions(asl_a_positions)
    time.sleep(0.02)  # Pause to view the letter 'A'
    
    # Revert to default position
    set_joint_positions(asl_revert, steps=200, delay=0.01)

# Convert lists to NumPy arrays
expert_states = np.array(expert_states)
expert_actions = np.array(expert_actions)

# Save to disk
np.save('expert_states_a.npy', expert_states)
np.save('expert_actions_a.npy', expert_actions)

print("Expert demonstrations saved successfully.")

# Keep the simulation running to view the result
while True:
    viewer.render()
    time.sleep(0.01)
