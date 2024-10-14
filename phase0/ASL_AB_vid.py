import mujoco
import numpy as np
import os
import time
import imageio

# Load the model
xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Offscreen rendering setup
width, height = 640, 480
options = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=10000)

# Try to use an offscreen context for rendering
gl_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150, offscreen=True)

# Create a video writer
video_filename = "render_pose_A.mp4"
fps = 30
writer = imageio.get_writer(video_filename, fps=fps)

# Function to render and save frames
def render_and_save_frame():
    img = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), data, gl_context)
    writer.append_data(img)

# Function to set joint positions and render
def set_joint_positions(joint_positions, steps=200, delay=0.02):
    for i in range(steps):
        for joint, target_position in joint_positions.items():
            current_position = data.ctrl[joint]
            new_position = current_position + (target_position - current_position) * (i / steps)
            data.ctrl[joint] = new_position
        mujoco.mj_step(model, data)

        # Render the frame and save to the video
        render_and_save_frame()
        time.sleep(delay)

finger_actuators = {
    "ForeFinger": [model.actuator(name).id for name in [
        "ForeFingerJoint0_act", "ForeFingerJoint1_act", "ForeFingerJoint2_act", "ForeFingerJoint3_act"]],
    "MiddleFinger": [model.actuator(name).id for name in [
        "MiddleFingerJoint0_act", "MiddleFingerJoint1_act", "MiddleFingerJoint2_act", "MiddleFingerJoint3_act"]],
    "RingFinger": [model.actuator(name).id for name in [
        "RingFingerJoint0_act", "RingFingerJoint1_act", "RingFingerJoint2_act", "RingFingerJoint3_act"]],
    "LittleFinger": [model.actuator(name).id for name in [
        "LittleFingerJoint0_act", "LittleFingerJoint1_act", "LittleFingerJoint2_act", "LittleFingerJoint3_act"]],
    "Thumb": [model.actuator(name).id for name in [
        "ThumbJoint0_act", "ThumbJoint1_act", "ThumbJoint2_act", "ThumbJoint3_act", "ThumbJoint4_act"]],
}

# Joint positions for the letter 'A' in ASL
asl_a_positions = {
    **{actuator: 1.6 for finger in ["ForeFinger", "MiddleFinger", "RingFinger", "LittleFinger"] for actuator in finger_actuators[finger]},
    finger_actuators["ForeFinger"][3]: -0.01,
    finger_actuators["MiddleFinger"][3]: -0.01,
    finger_actuators["RingFinger"][3]: -0.01,
    finger_actuators["LittleFinger"][3]: -0.01,
    finger_actuators["Thumb"][0]: -0.9,
    finger_actuators["Thumb"][1]: 0.0,
    finger_actuators["Thumb"][2]: 0.262,
    finger_actuators["Thumb"][3]: 0.5,
    finger_actuators["Thumb"][4]: 0.5,
}

# Simulate and render the pose 'A'
set_joint_positions(asl_a_positions)

# Close the video writer
writer.close()
print(f"Video saved as {video_filename}")



