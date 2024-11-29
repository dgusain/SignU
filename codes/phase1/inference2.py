import os
import time
import torch
import mujoco_py
from stable_baselines3 import PPO
import numpy as np

# Define the path to your MuJoCo XML model and trained PPO model
xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')
model_path = '/home/ducky/Downloads/Bot_hand/hand_pose_ppo_gpu_1.zip'

# Load MuJoCo model
try:
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    print("MuJoCo model loaded successfully.")
except Exception as e:
    print(f"Failed to load MuJoCo model: {e}")
    exit(1)

# Load the PPO model
try:
    print("Loading PPO model weights...")
    ppo_model = PPO.load(model_path)
    print("PPO model loaded successfully.")
except Exception as e:
    print(f"Failed to load PPO model: {e}")
    exit(1)

# Define actuator ranges
actuator_ranges = sim.model.actuator_ctrlrange  # Shape: (num_actuators, 2)
actuator_min = actuator_ranges[:, 0]
actuator_max = actuator_ranges[:, 1]

# Define ground truth quaternions
ground_truth_quats = np.array([
    [6.32658406e-04,  1.25473697e-03, -9.99718591e-03,  1.56702220e+00],
    [1.25473697e-03, -9.99718591e-03,  1.56702220e+00,  1.56730258e+00],
    [-0.00999719,  1.5670222,   1.56730258,  1.5674143],
    [1.5670222,   1.56730258,  1.5674143,  -0.00999801],
    [1.56730258,  1.5674143,  -0.00999801,  1.56701926],
    [1.5674143,  -0.00999801,  1.56701926,  1.56730194],
    [-0.00999801,  1.56701926,  1.56730194,  1.5674143],
    [1.56701926,  1.56730194,  1.5674143,  -0.00999756],
    [1.56730194,  1.5674143,  -0.00999756,  1.56701717],
    [1.5674143,  -0.00999756,  1.56701717,  1.56730177],
    [-0.00999756,  1.56701717,  1.56730177,  1.56741429],
    [1.56701717, 1.56730177, 1.56741429, 0.00868252],
    [1.56730177,  1.56741429,  0.00868252, -0.01000805],
    [1.56741429,  0.00868252, -0.01000805,  1.56708349],
    [0.00868252, -0.01000805,  1.56708349,  1.56730911],
    [-0.01000805,  1.56708349,  1.56730911,  1.56741508],
    [1.56708349, 1.56730911, 1.56741508, 0.49916711],
    [1.56730911, 1.56741508, 0.49916711, 0.50022545],
    [1.56741508, 0.49916711, 0.50022545, 0.25330455],
    [4.99167109e-01, 5.00225452e-01, 2.53304550e-01, 4.08440608e-04],
    [5.00225452e-01,  2.53304550e-01,  4.08440608e-04, -9.00000689e-01],
    [2.53304550e-01,  4.08440608e-04, -9.00000689e-01,  1.00000000e-01],
    [4.08440608e-04, -9.00000689e-01,  1.00000000e-01, -1.00000000e-01],
    [-0.90000069,  0.1,        -0.1,         0.01463168],
    [0.1,        -0.1,         0.01463168,  1.0]
], dtype=np.float32)

def normalize_quaternion(q):
    """
    Normalize each quaternion to ensure it's a unit quaternion.
    
    Args:
        q (np.ndarray): Array of quaternions with shape (num_joints, 4).
    
    Returns:
        np.ndarray: Normalized quaternions.
    """
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.maximum(norm, 1e-8)  # Prevent division by zero

def calculate_confidence(state, ground_truth_quats, target_threshold=90):
    """
    Calculate the average confidence score based on quaternion similarity.
    
    Args:
        state (np.ndarray): Current state from the simulation (first 24 qpos).
        ground_truth_quats (np.ndarray): Ground truth quaternions.
        target_threshold (float): Threshold for determining if the target pose is reached.
    
    Returns:
        float: Average confidence score.
    """
    rendered_quat = []
    num_joints = state.shape[0] // 4  # Assuming qpos has joint quaternions
    for joint in range(num_joints):
        q = state[joint*4 : joint*4 + 4]
        rendered_quat.append(q)
    rendered_quat = np.array(rendered_quat, dtype=np.float32)
    rendered_quat = normalize_quaternion(rendered_quat)
    gt_quat_normalized = normalize_quaternion(ground_truth_quats)
    similarity = np.abs(np.sum(rendered_quat * gt_quat_normalized, axis=1)) * 100
    avg_confidence = np.mean(similarity)
    return avg_confidence

def run_episode(sim, viewer, ppo_model, ground_truth_quats, actuator_min, actuator_max, target_threshold=90, max_steps=100):
    """
    Run a single episode of simulation using the PPO model and accumulate rewards.
    
    Args:
        sim (mujoco_py.MjSim): The MuJoCo simulation instance.
        viewer (mujoco_py.MjViewer): The MuJoCo viewer instance.
        ppo_model (stable_baselines3.PPO): The trained PPO model.
        ground_truth_quats (np.ndarray): Ground truth quaternions for confidence calculation.
        actuator_min (np.ndarray): Minimum actuator control ranges.
        actuator_max (np.ndarray): Maximum actuator control ranges.
        target_threshold (float): Threshold for confidence to determine if the episode is done.
        max_steps (int): Maximum number of steps per episode.
    
    Returns:
        float: Total accumulated reward for the episode.
    """
    total_reward = 0.0
    steps_taken = 0
    done = False

    # Initial observation
    obs = sim.data.qpos[:24].astype(np.float32)

    while not done and steps_taken < max_steps:
        # Use PPO model to predict actions
        action, _ = ppo_model.predict(obs, deterministic=True)
        
        # Rescale action from [-1, 1] to actuator control range
        rescaled_action = actuator_min + (action + 1) * (actuator_max - actuator_min) / 2
        sim.data.ctrl[:] = rescaled_action
        sim.step()  # Advance the simulation

        # Update observation
        obs = sim.data.qpos[:24].astype(np.float32)

        # Calculate confidence
        confidence = calculate_confidence(sim.data.qpos[:24], ground_truth_quats, target_threshold)
        
        # Determine if the episode is done
        if confidence >= target_threshold:
            reward = confidence - 50  # Reward for reaching target pose
            done = True
        else:
            reward = -1.0  # Penalty for taking extra steps

        # Accumulate total reward
        total_reward += reward

        # Render the simulation
        viewer.render()

        steps_taken += 1

    return total_reward

# Run inference for 5 episodes
for episode in range(5):
    print(f"\nRunning episode {episode + 1}...")

    # Run the episode and get total reward
    total_reward = run_episode(sim, viewer, ppo_model, ground_truth_quats, actuator_min, actuator_max)
    print(f"Episode {episode + 1} completed. Total Reward: {total_reward:.2f}")
    
    time.sleep(2)
    
    # Reset the simulation after each episode
    sim.reset()
    viewer.render()  # Optionally render after reset
    time.sleep(1)  # Short pause between episodes

print("\nInference complete.")
