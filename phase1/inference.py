import os
import time
import torch
import mujoco_py
from stable_baselines3 import PPO
import numpy as np

# Define the path to your MuJoCo XML model and trained PPO model
xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')
model_path = '/home/ducky/Downloads/Bot_hand/hand_pose_ppo_gpu_0.zip'

# Load MuJoCo model
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Load the PPO model
print("Loading model weights:")
ppo_model = PPO.load(model_path)

# Function to render simulation for one episode
def run_episode(sim, viewer, ppo_model):
    obs = sim.data.qpos[:24].astype(np.float32)  # Initial observation (first 24 joint positions)
    
    for _ in range(100):  # Specify the number of steps per episode
        # Use PPO model to predict actions
        action, infor = ppo_model.predict(obs)
        
        # Rescale action from [-1, 1] to actuator control range
        sim.data.ctrl[:] = action
        sim.step()  # Advance the simulation
        
        # Update observation
        obs = sim.data.qpos[:24].astype(np.float32)
        
        # Render the simulation
        viewer.render()

# Run inference for 5 episodes
for episode in range(5):
    print(f"Running episode {episode + 1}...")
    
    # Run the episode
    run_episode(sim, viewer, ppo_model)
    time.sleep(2)
    
    # Reset the simulation after each episode
    sim.reset()
    time.sleep(1)  # Short pause between episodes
    viewer.render()

print("Inference complete.")
