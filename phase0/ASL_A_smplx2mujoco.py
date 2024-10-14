# This file is hardcoded for letter A. The code is not smart enough to know when to pull or release the actuator. It has been hardcoded to release the actuator. It needs to be coded to handle when to pull, for example, in letter B. Further, it needs to learn how to handle the cases of the thumb and little finger. 
import json
import os
import time
import numpy as np
import mujoco_py

# ------------------------ Configuration ------------------------

# Path to the SMPLX JSON file
SMPLX_JSON_PATH = '/home/ducky/Downloads/ASL_Fei/json_label_A/031.json'  # Replace with your JSON file path

# Path to the MuJoCo model XML file
MUJOCO_XML_PATH = '/home/ducky/Downloads/Bot_hand/bot_hand.xml'  # Replace with your MuJoCo XML file path

# Scaling factors for actuator controls (to be calibrated)
# These values may need adjustment based on your specific model and calibration
PITCH_SCALE = 1.8  # Flexion/Extension scaling
YAW_SCALE = 0.8    # Abduction/Adduction scaling
ROLL_SCALE = 0.0   # Rotation scaling

# ------------------------ Mapping Functions ------------------------

def scale_angle(angle, scale_factor):
    range_min, range_max = -1, 0
 
    # Normalize the angle to be between -pi and pi
    angle = np.clip(angle, -np.pi, np.pi)

    # Scale angle to actuator position range
    normalized_angle = (angle + np.pi) / (2 * np.pi)  # Normalizes angle from [0, 1]
    actuator_position = range_min + (1-normalized_angle) * (range_max - range_min)

    # Clip actuator position to the range
    actuator_position = np.clip(actuator_position, range_min, range_max)
    return actuator_position * scale_factor

def map_joint_angles_to_tendons(smplx_joint_angles, tendon_actuators):
    tendon_ctrls = {}

    tendon_to_joint_mapping = {
        "FMCP":"F3",
        "FPIP":"F2",
        "FDIP":"F1",
        "MMCP":"M3",
        "MPIP":"M2",
        "MDIP":"M1",
        "RMCP":"R3",
        "RPIP":"R2",
        "RDIP":"R1",
        "LMCP":"L3",
        "LPIP":"L2",
        "LDIP":"L1",
        "TMCP":"T3",
        "TPIP":"T2",
        "TDIP":"T1",
    }

    # Map each joint's Pitch to corresponding pull/release actuators
    for finger in ["ForeFinger", "MiddleFinger", "RingFinger", "LittleFinger", "Thumb"]:
        for joint_level in ["MCP", "PIP", "DIP"]:
            
            joint_key = f"{finger[0]}{joint_level}"  # e.g., "F1", "M1", etc.
            jk = tendon_to_joint_mapping[joint_key]
            
            angles = smplx_joint_angles.get(jk, {"Pitch": 0.0, "Yaw": 0.0, "Roll": 0.0}) ###
            
            if finger == "LittleFinger": 
                x = 2
            elif finger == "Thumb":
                x = 2
            else:
                x = 0
     
            if joint_level == "MCP":
                pull_actuator = tendon_actuators[finger][0+x]  # e.g., FFJ3r_motor
                release_actuator = tendon_actuators[finger][1+x]  # e.g., FFJ3l_motor
                move_angle = angles["Yaw"]
                movement = scale_angle(move_angle,YAW_SCALE)
            else:
                move_angle = angles["Pitch"]
                movement = scale_angle(move_angle,PITCH_SCALE)
            
                
            if joint_level == "PIP":
                pull_actuator = tendon_actuators[finger][2+x]  # e.g., FFJ2u_motor
                release_actuator = tendon_actuators[finger][3+x]  # e.g., FFJ2d_motor
            elif joint_level == "DIP":
                pull_actuator = tendon_actuators[finger][4+x]  # e.g., FFJ1u_motor
                release_actuator = tendon_actuators[finger][5+x]  # e.g., FFJ1d_motor
                       
            print(f"Finger: {finger}, Actuator: {pull_actuator},{release_actuator}, joint level: {joint_level}, Angle: {move_angle}, Movement: {movement}")
                             
            #if val1 > 0: 
             #   tendon_ctrls[pull_actuator] = movement
              #  tendon_ctrls[release_actuator] = 0
            #else:
            # hardcoded to release the actuator. 
            # code must be smart enough to know when to release the tendon and when to pull. 
            tendon_ctrls[pull_actuator] = 0
            tendon_ctrls[release_actuator] = movement


    return tendon_ctrls

def map_wrist_angles_to_tendons(wrist_angles, tendon_actuators):
    tendon_ctrls = {}

    # Map Pitch
    pitch_angle = wrist_angles.get("Pitch",0.0)
    print(f"Pitch angle: {pitch_angle}")
    pitch = scale_angle(wrist_angles.get("Pitch", 0.0), PITCH_SCALE)
    print(f"Pitch actuator position: {pitch}")
    
    tendon_ctrls[tendon_actuators["Wrist"][0]] = pitch     # WRJ1r_motor (pull)
    tendon_ctrls[tendon_actuators["Wrist"][1]] = -pitch    # WRJ1l_motor (release)

    # Map Yaw
    yaw = scale_angle(wrist_angles.get("Yaw", 0.0), YAW_SCALE)
    tendon_ctrls[tendon_actuators["Wrist"][2]] = yaw       # WRJ0u_motor (pull)
    tendon_ctrls[tendon_actuators["Wrist"][3]] = -yaw      # WRJ0d_motor (release)

    return tendon_ctrls

# ------------------------ Main Function ------------------------

def main():
    # -------------------- Load SMPLX JSON Data --------------------
    if not os.path.exists(SMPLX_JSON_PATH):
        print(f"Error: SMPLX JSON file not found at {SMPLX_JSON_PATH}")
        return

    with open(SMPLX_JSON_PATH, 'r') as f:
        smplx_data = json.load(f)

    # Extract right_hand_pose and body_pose
    right_hand_pose = smplx_data.get("right_hand_pose", [[]])[0]  # Assuming single entry
    #print(right_hand_pose)
    body_pose = smplx_data.get("body_pose", [[]])[0]            # Assuming single entry

    # -------------------- Load MuJoCo Simulation --------------------
    if not os.path.exists(MUJOCO_XML_PATH):
        print(f"Error: MuJoCo XML file not found at {MUJOCO_XML_PATH}")
        return

    model = mujoco_py.load_model_from_path(MUJOCO_XML_PATH)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)

    # -------------------- Define Tendon Actuators Mapping --------------------
    # Ensure the model is loaded before mapping actuators
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

    # -------------------- Parse Right Hand Joints --------------------
    # Define joint keys and their indices
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
    #print(smplx_joint_angles)

    # -------------------- Parse Right Wrist Angles --------------------
    # SMPLX body_pose standard ordering:
    # Joint Index 13 corresponds to Right Wrist
    # Each joint has 3 values: Pitch, Yaw, Roll
    # Ensure body_pose has enough elements
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

    # -------------------- Map Joint Angles to Tendon Controls --------------------
    # Map finger joint angles
    finger_tendon_ctrls = map_joint_angles_to_tendons(smplx_joint_angles, tendon_actuators)
    
    # Map wrist angles
    #wrist_tendon_ctrls = map_wrist_angles_to_tendons(wrist_angles, tendon_actuators)

    # Combine all tendon controls
    #total_tendon_ctrls = {**finger_tendon_ctrls, **wrist_tendon_ctrls}
    total_tendon_ctrls = finger_tendon_ctrls

    # -------------------- Apply Tendon Controls Smoothly --------------------
    def set_tendon_controls(sim, viewer, target_ctrls, steps=200, delay=0.02):        
        for i in range(steps):
            for actuator_id, target_value in target_ctrls.items():
                current_value = sim.data.ctrl[actuator_id]
                # Linear interpolation
                new_value = current_value + (target_value - current_value) * (i / steps)
                sim.data.ctrl[actuator_id] = new_value
            sim.step()
            viewer.render()
            time.sleep(delay)
     
    # Apply the mapped tendon controls
    set_tendon_controls(sim, viewer, total_tendon_ctrls)
    time.sleep(0.1)

    # -------------------- Keep the Simulation Running --------------------
    print("Simulation running. Press Ctrl+C to exit.")
    try:
        while True:
            sim.step()
            viewer.render()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Simulation terminated.")

# ------------------------ Execute Main Function ------------------------

if __name__ == "__main__":
    main()

