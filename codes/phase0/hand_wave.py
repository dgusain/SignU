import mujoco_py
import os
import time

# Load the model
xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Actuator indices for the wrist joint based on the uploaded actuators_for_joints.xml
wrist_actuator_id = sim.model.actuator_name2id("WristJoint1_act")

# Function to wave the hand left and right
def wave_hand(wrist_actuator_id, waves=5):
    for _ in range(waves):
        # Move wrist to the left
        for i in range(100):  # Adjust the range for smoother movement
            sim.data.ctrl[wrist_actuator_id] = -0.52 + i * 0.0069  # Increment control value within the joint's range
            sim.step()
            viewer.render()
            time.sleep(0.01)
        
        # Move wrist to the right
        for i in range(100):  # Adjust the range for smoother movement
            sim.data.ctrl[wrist_actuator_id] = 0.17 - i * 0.0069  # Decrement control value within the joint's range
            sim.step()
            viewer.render()
            time.sleep(0.01)

# Main loop to wave the hand
wave_hand(wrist_actuator_id, waves=5)


'''
Notes: 
1. -0.52 and 0.17 is the range that i have defined in my actuators_for_joints.xml file for this WristJoint1_act. 
2. In order to move between this range using a 100 steps, we perform basic math: 
Total range = (0.17 - (-0.52)) = 0.69
Total range of each step = 0.69/100 = 0.0069
i = current position of actuator.

3. The more the number of steps for the range, the smoother the motion. 
'''

