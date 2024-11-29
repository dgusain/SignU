import json
import os
import time
import numpy as np
import mujoco_py

SMPLX_JSON_PATH = '/home/ducky/Downloads/ASL_Fei/json_label_A/031.json'
MUJOCO_XML_PATH = '/home/ducky/Downloads/Bot_hand/bot_hand.xml' 

def main():
    # -------------------- Load SMPLX JSON Data --------------------
    if not os.path.exists(SMPLX_JSON_PATH):
        print(f"Error: SMPLX JSON file not found at {SMPLX_JSON_PATH}")
        return

    with open(SMPLX_JSON_PATH, 'r') as f:
        smplx_data = json.load(f)

    # Extract right_hand_pose and body_pose
    right_hand_pose = smplx_data.get("right_hand_pose", [[]])[0]  
    print(right_hand_pose)
    body_pose = smplx_data.get("body_pose", [[]])[0]
    
    # -------------------- Load MuJoCo Simulation --------------------
    if not os.path.exists(MUJOCO_XML_PATH):
        print(f"Error: MuJoCo XML file not found at {MUJOCO_XML_PATH}")
        return

    model = mujoco_py.load_model_from_path(MUJOCO_XML_PATH)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)               

    tendon_actuators = {
        "Wrist": [ 
            model.actuator_name2id(name) 
            for name in [
                "WRJ1r_motor", "WRJ1l_motor", "WRJ0u_motor", "WRJ0d_motor"]
        ],
        "ForeFinger": [ 
            model.actuator_name2id(name) 
            for name in [
                "FFJ3r_motor", "FFJ3l_motor", "FFJ2u_motor", "FFJ2d_motor", 
                "FFJ1u_motor", "FFJ1d_motor"]
        ],
        "MiddleFinger": [ 
            model.actuator_name2id(name) 
            for name in [
                "MFJ3r_motor", "MFJ3l_motor", "MFJ2u_motor", "MFJ2d_motor", 
                "MFJ1u_motor", "MFJ1d_motor"]
        ],
        "RingFinger": [ 
            model.actuator_name2id(name) 
            for name in [
                "RFJ3r_motor", "RFJ3l_motor", "RFJ2u_motor", "RFJ2d_motor", 
                "RFJ1u_motor", "RFJ1d_motor"]
        ],
        "LittleFinger": [ 
            model.actuator_name2id(name) 
            for name in [
                "LFJ4u_motor", "LFJ4d_motor", "LFJ3r_motor", "LFJ3l_motor", 
                "LFJ2u_motor", "LFJ2d_motor", "LFJ1u_motor", "LFJ1d_motor"]
        ],
        "Thumb": [ 
            model.actuator_name2id(name) 
            for name in [
                "THJ4a_motor", "THJ4c_motor", "THJ3u_motor", "THJ3d_motor", 
                "THJ2u_motor", "THJ2d_motor", "THJ1r_motor", "THJ1l_motor", 
                "THJ0r_motor", "THJ0l_motor"]
        ],
    }

    smplx_joints = {
        "F1": {"Pitch": 0, "Yaw": 1, "Roll": 2},   # Index MCP
        "F2": {"Pitch": 3, "Yaw": 4, "Roll": 5},   # Index PIP
        "F3": {"Pitch": 6, "Yaw": 7, "Roll": 8},   # Index DIP
        "M1": {"Pitch": 9, "Yaw": 10, "Roll": 11}, # Middle MCP
        "M2": {"Pitch": 12, "Yaw": 13, "Roll": 14},# Middle PIP
        "M3": {"Pitch": 15, "Yaw": 16, "Roll": 17},# Middle DIP
        "R1": {"Pitch": 18, "Yaw": 19, "Roll": 20},# Ring MCP
        "R2": {"Pitch": 21, "Yaw": 22, "Roll": 23},# Ring PIP
        "R3": {"Pitch": 24, "Yaw": 25, "Roll": 26},# Ring DCP
        "L1": {"Pitch": 27, "Yaw": 28, "Roll": 29},# Little MCP
        "L2": {"Pitch": 30, "Yaw": 31, "Roll": 32},# Little PIP
        "L3": {"Pitch": 33, "Yaw": 34, "Roll": 35},# Little DIP
        "T1": {"Pitch": 36, "Yaw": 37, "Roll": 38},# Thumb MCP
        "T2": {"Pitch": 39, "Yaw": 40, "Roll": 41},# Thumb PIP
        "T3": {"Pitch": 42, "Yaw": 43, "Roll": 44},# Thumb DIP
    }

    # Create a dictionary to hold joint angles
    smplx_joint_angles = {}
    for idx, joint_key in enumerate(smplx_joints.keys()):
    # Calculate the starting index for the current joint (every joint has 3 values)
        start_idx = idx * 3
    
    # Extract the pitch, yaw, and roll values
        pitch = right_hand_pose[start_idx]
        yaw = right_hand_pose[start_idx + 1]
        roll = right_hand_pose[start_idx + 2]
    
    # Print the extracted values for debugging
        #print(f"{joint_key} - Pitch: {pitch}, Yaw: {yaw}, Roll: {roll}")
    
    # Store the extracted values in the smplx_joint_angles dictionary
        smplx_joint_angles[joint_key] = {
            "Pitch": pitch,
            "Yaw": yaw,
            "Roll": roll
        }
    print(smplx_joint_angles)
    
    # Wrist indexing
    if len(body_pose) < 14 * 3:
        print("Error: body_pose array does not contain enough elements for Right Wrist.")
        return

    right_wrist_index = 13  # 0-based indexing
    wrist_pitch = body_pose[right_wrist_index * 3]
    wrist_yaw = body_pose[right_wrist_index * 3 + 1]
    wrist_roll = body_pose[right_wrist_index * 3 + 2]

    wrist_angles = {
        "Pitch": wrist_pitch,
        "Yaw": wrist_yaw,
        "Roll": wrist_roll
    }
    
    

main()

