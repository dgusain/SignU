import mujoco_py
import os
import time
from mujoco_py import GlfwContext, MjRenderContextOffscreen

# Ensure the viewer context is initialized
GlfwContext(offscreen=True)

# Load the model
xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
viewer = MjRenderContextOffscreen(sim, device_id=0)

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

# Function to set joint positions and capture frames
def set_joint_positions(joint_positions, steps=200, delay=0.02, video_writer=None):
    for i in range(steps):
        for joint, target_position in joint_positions.items():
            current_position = sim.data.ctrl[joint]
            new_position = current_position + (target_position - current_position) * (i / steps)
            sim.data.ctrl[joint] = new_position
        sim.step()
        viewer.render()
        if video_writer:
            video_writer.capture_frame()
        time.sleep(delay)

# Joint positions for the letter 'A' in ASL
asl_a_positions = {
    **{actuator: 1.6 for finger in ["ForeFinger", "MiddleFinger", "RingFinger", "LittleFinger"] for actuator in finger_actuators[finger]},
    finger_actuators["Thumb"][0]: -0.9,  # ThumbJoint0_act
    finger_actuators["Thumb"][1]: 0.0,   # ThumbJoint1_act
    finger_actuators["Thumb"][2]: 0.262, # ThumbJoint2_act
    finger_actuators["Thumb"][3]: 0.5,   # ThumbJoint3_act
    finger_actuators["Thumb"][4]: 0.5,   # ThumbJoint4_act
}

# Joint positions for the letter 'B' in ASL
asl_b_positions = {
    **{actuator: 0.0 for finger in ["LittleFinger","RingFinger", "Thumb","MiddleFinger", "ForeFinger"] for actuator in finger_actuators[finger]},
    finger_actuators["Thumb"][0]: 0.0,   # ThumbJoint0_act
    finger_actuators["Thumb"][1]: -0.5,  # ThumbJoint1_act
    finger_actuators["Thumb"][2]: 0.262, # ThumbJoint2_act
    finger_actuators["Thumb"][3]: 1.309, # ThumbJoint3_act
    finger_actuators["Thumb"][4]: 1.047, # ThumbJoint4_act
}

asl_revert = {actuator: 0.0 for finger in ["Thumb"] for actuator in finger_actuators[finger]}

# Setup movie writer
video_path = '/path/to/save/simulation.mp4'
video_writer = viewer.start_recording(video_path)

for i in range(2): 
    # Set the hand to display the letter 'A' and capture frames
    set_joint_positions(asl_a_positions, video_writer=video_writer)
    time.sleep(0.02)  # Pause to view the letter 'A'
    
    # Set the hand to display the letter 'B' and capture frames
    set_joint_positions(asl_b_positions, video_writer=video_writer)
    time.sleep(0.02)  # Pause to view the letter 'B'
    
    # Revert the hand to the initial position and capture frames
    set_joint_positions(asl_revert, 200, 0.01, video_writer=video_writer)

# Stop recording and save the simulation to a .mp4 file
viewer.stop_recording()

# Keep the simulation running to view the result
while True:
    viewer.render()
    time.sleep(0.01)

