
# Project: Reinforcement Learning for American Sign Language Fingerspelling in MuJoCo

## Context
This project leverages reinforcement learning to teach an agent to form various hand gestures for fingerspelling in American Sign Language (ASL). The agent learns to form letters like "A", "B", etc., autonomously. The primary challenge is mapping SMPLX joint data to MuJoCo for physically plausible gesture rendering, as MuJoCo accounts for realistic physics unlike SMPLX where finger collisions can occur.

### Roadmap:
- **Phase 1**: Dataset preparation and integration with RL framework (Fall 2024)
- **Phase 2**: Model training and inference (Spring 2025)

### Challenges:
1. **Data Collection**: Gathering SMPLX modeled data from input videos of ASL, to be used as ground truth.
2. **MuJoCo Model**: Developing a MuJoCo model capable of rendering hand poses (starting with the right hand, using the ShadowHand skeleton).
3. **Mapping Pipeline**: Creating a mapping from SMPLX's 15 joints to MuJoCo's 24 joints for accurate pose rendering.
4. **Reinforcement Learning**: Training the agent to form hand gestures in MuJoCo using reinforcement learning.

---

## Summary of Approaches

### **9th August: Initial Mapping Attempts**
- **Blender Setup**: Installed Blender 2.92 and worked with SMPLX addon.
- **Mapping Strategy**: Used proximal, middle, and distal joints from SMPLX.
  - **Issue**: SMPLX did not load knuckles correctly; mapping was incomplete.
  - **Fix**: Addressed errors in the rendering process but found the mapping still problematic.

- **Comparison Strategy**: Collaborated with Fei, comparing 3D coordinate rendering from Blender to MuJoCo for mapping corrections.
  
### **26th August: Blender and Mapping Refinement**
- Converted SMPLX `.pkl` files to `.json` and rendered poses in MuJoCo.
  - **Issue**: SMPLX model did not account for distal joints.
  - **Fix**: Updated mapping for fingers (Proximal → Knuckle, Middle → Proximal, Distal → Middle).

- **Realization**: Rotations were provided in XYZ Euler angles, not axis-angles. Two possible fixes:
  1. Convert Euler angles to axis-angles in Blender script.
  2. Modify rendering script to handle XYZ Euler angles.

### **26th August: New Scripts and Improvements**
- Created Python script `smplx_pkl2bot.py` to convert SMPLX `.pkl` to MuJoCo joint positions.
- Developed `smplx_mapped2render.py`, using KPM approach and interpolation for distal joint estimation.
  - **Outcome**: Partial success, but not perfect.

### **27th August: Further Mapping Refinements**
- Implemented a revised mapping strategy (v2), using x and y rotations for finger joints.
  - **Key Insight**: SMPLX rotations are not cumulative but absolute relative to the base pose, simplifying frame-by-frame processing.

### **30th September: MuJoCo Rendering Refinements**
- Rendered ASL letter "A" in MuJoCo but observed hand falling due to simulation continuing.
  - **Fix**: Stopped the simulation to preserve the pose.

### **3rd-4th October: Tendon-Based Mapping**
- Developed a tendon-based strategy for SMPLX to MuJoCo mapping, with detailed actuator mappings for each joint.
- Suggested making the tendon mapping a learnable parameter through reinforcement learning (RL).

### **Reinforcement Learning Progress**
- Multi-GPU reinforcement learning setup for training the agent to render ASL letter "A".
  - **Results**: Rewards ranged between 41 and 46 over 100 iterations using multi-GPU setups.

---

## Next Steps:
- **Phase 1**: Complete dataset integration and mapping pipeline.
- **Phase 2**: Fine-tune RL model and perform inference for fingerspelling in ASL.


## Reinforcement Learning Phase Progress

This project involves simulating and controlling a robotic hand using the MuJoCo physics engine integrated with OpenAI Gym environments. The primary objective is to train a policy using Proximal Policy Optimization (PPO) to align the simulated hand's pose with predefined or dynamically loaded ground truth quaternions. The project progresses through two phases:

### Phases
#### Phase 1: Hardcoded Ground Truth Quaternions
In the initial phase, the environment uses a predefined list of quaternions as ground truth. These quaternions represent the desired orientations for each joint in the robotic hand, serving as targets for the reinforcement learning agent to achieve.

#### Phase 2: Dynamic Ground Truth from SMPL-X JSON Files
The second phase enhances flexibility by sourcing ground truth quaternions from SMPL-X JSON files. This allows for dynamic loading of different poses without modifying the codebase, facilitating scalability and adaptability to various hand poses.

### Code Structure
#### Environment Class (HandEnv)
The `HandEnv` class inherits from `gym.Env` and encapsulates the simulation environment, including state management, action processing, and reward computation.

**Initialization**:
- **Model Loading**: Loads the MuJoCo model from a specified XML file.
- **Simulation Setup**: Initializes the MuJoCo simulator (`MjSim`) with the loaded model.
- **Observation and Action Spaces**:
  - **Observation Space**: 24-dimensional continuous space.
  - **Action Space**: Based on actuator control ranges, normalized between -1 and 1.
- **Ground Truth Quaternions**:
  - **Phase 1**: Hardcoded list.
  - **Phase 2**: Dynamically loaded from SMPL-X JSON files.

#### Step Function
Processes the agent's action by:
- **Action Rescaling**: Transforms normalized actions back to actuator-specific ranges.
- **Simulation Step**: Applies the rescaled actions and advances the simulation.
- **Reward Calculation**: Computes reward based on quaternion similarity.
- **Termination Condition**: Ends the episode if confidence score exceeds a threshold or maximum steps reached.

### Reward and Confidence Calculation
- **Confidence Score**: Measures similarity between rendered and ground truth quaternions.
- **Reward**: Based on the average confidence score.

### Key Features
- **Dynamic Ground Truth Loading**: Allows switching from hardcoded quaternions to loading from SMPL-X JSON files.
- **Efficient Training Pipeline**: Utilizes RecurrentPPO with optimized parameters for policy learning.
- **Comprehensive Logging and Visualization**: Tracks and plots training rewards for performance assessment.

### Installation
```bash
git clone https://github.com/yourusername/robotic-hand-simulation.git
cd robotic-hand-simulation
```

#### Create a Virtual Environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies:
```bash
pip install -r requirements.txt
```

Ensure `requirements.txt` includes necessary packages such as `gymnasium`, `mujoco_py`, `stable_baselines3`, `sb3_contrib`, `torch`, `numpy`, and `matplotlib`.

#### Setup MuJoCo:
1. Download MuJoCo binaries and models.
2. Set environment variables:
```bash
export MUJOCO_PY_MUJOCO_PATH=/path/to/mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/mujoco/bin
```

### Usage
1. Prepare SMPL-X JSON Files containing `right_hand_pose`.
2. Update paths in the code to point to the MuJoCo XML model and SMPL-X JSON file.
3. Run Training:
```bash
python your_training_script.py
```

### Troubleshooting
- **MuJoCo Model Loading Errors**: Ensure the XML path is correct and MuJoCo is properly installed.
- **JSON Parsing Issues**: Validate the JSON structure, ensuring the `right_hand_pose` key exists.
- **Dependency Conflicts**: Use `requirements.txt` for compatible versions, or consider Docker for consistency.
- **CUDA and GPU Issues**: Verify CUDA installation, GPU availability, and proper configuration.
