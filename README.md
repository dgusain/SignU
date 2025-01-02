
# Project: Reinforcement Learning for American Sign Language Fingerspelling in MuJoCo

## Context
This project leverages reinforcement learning to teach an agent to form various hand gestures for fingerspelling in American Sign Language (ASL). The agent learns to form letters like "A", "B", etc., autonomously. The primary challenge is mapping SMPLX joint data to MuJoCo for physically plausible gesture rendering, as MuJoCo accounts for realistic physics unlike SMPLX where finger collisions can occur.
<table>
<tr>
  <td width="25%">
     <img src="https://github.com/dgusain/SignU/blob/main/ASL_VW_mj_git.gif" alt="SignMimic" width="250" height="200">  
  </td>
    <td width="25%">
     <img src="https://github.com/dgusain/SignU/blob/main/ASL_AB_mj_git.gif" alt="SignMimic" width="250" height="200">  
  </td>
</tr>
</table>

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


## Reinforcement Learning Phase Progress

This project involves simulating and controlling a robotic hand using the MuJoCo physics engine integrated with OpenAI Gym environments. The primary objective is to train a policy using Proximal Policy Optimization (PPO) to align the simulated hand's pose with predefined or dynamically loaded ground truth quaternions. The project progresses through two phases:

### Phase 1: Hardcoded Ground Truth Quaternions
In the initial phase, the environment uses a predefined list of quaternions as ground truth. These quaternions represent the desired orientations for each joint in the robotic hand, serving as targets for the reinforcement learning agent to achieve.

The reinforcement learning process aims to train a robotic hand model to replicate specific hand poses, such as forming the letter "A" in American Sign Language. Below is a detailed breakdown of the key components and processes used in the project:

#### 1. Environment Setup (HandEnv Class)
   - **Observation and Action Spaces**:
     - The observation space consists of a 24-dimensional continuous space, representing joint positions.
     - The action space is normalized between `-1` and `1` and corresponds to actuator control inputs.
   - **Ground Truth Quaternions**:
     - In phase 1, a predefined list of quaternions serves as the ground truth, representing the desired hand pose (letter "A").
     - These quaternions are normalized and compared to the pose rendered by the simulation to assess similarity.

#### 2. Step Function
   - The `step` function processes the actions taken by the agent:
     - **Action Rescaling**: Actions are transformed from the normalized `[-1, 1]` range back to the actuator-specific control ranges.
     - **Simulation Step**: After applying the rescaled actions, the simulation advances by one step.
     - **State Extraction**: The new state (joint positions) is retrieved from the MuJoCo simulation.
     - **Reward Calculation**: The reward is based on how closely the simulated pose matches the ground truth quaternion (measured using a confidence score).
     - **Termination Condition**: An episode ends either when the confidence score exceeds a target threshold (95) or when a maximum number of steps (100) is reached.

#### 3. Reward and Confidence Calculation
   - **Confidence Score**: This score measures how similar the rendered pose's quaternions are to the ground truth. It is calculated by comparing the normalized quaternions of the rendered pose and the ground truth.
   - **Reward**: The reward for an episode is based on the average confidence score. The goal is to incentivize the agent to match the ground truth as closely as possible.

#### 4. Training Process
   - **RecurrentPPO**: The reinforcement learning algorithm used is **Proximal Policy Optimization (PPO)** with recurrent policies (LSTM). PPO is known for stable policy updates and works well in continuous action spaces like controlling a robotic hand.
   - **Environment**: Multiple parallel environments (via `DummyVecEnv`) are instantiated to speed up training.
   - **Callbacks**: A custom callback (`RewardCallback`) is implemented to track rewards during training. It logs the average rewards for every 100 episodes and stores these for later visualization.

#### 5. Training Parameters
   - **Total Timesteps**: The training process runs for a defined number of timesteps across the parallel environments.
   - **Learning Rate**: Set to `0.0001`, with other hyperparameters optimized for this task, including clipping (`0.4`) and entropy coefficient (`0.05`) for better exploration.
   - **State Dependent Exploration**: This feature is enabled to allow better exploration of the action space during training.

#### 6. Visualization
   - After training, the rewards are plotted to visualize the agent's learning progress. The plot displays the average reward per 100 episodes, helping to assess how well the model is learning to replicate the ground truth hand poses.

### Hyperparameters Overview

1. **Learning Rate (`learning_rate=0.0001`)**:
   - The learning rate controls how much the model's weights are adjusted during each update. A small learning rate (like `0.0001`) ensures that updates are incremental, reducing the risk of overshooting the optimal policy. However, it can make the training process slower.

2. **Clip Range (`clip_range=0.4`)**:
   - Proximal Policy Optimization (PPO) uses clipping to restrict the change in policy during training to prevent large updates that might destabilize the learning process. A clip range of `0.4` ensures that the policy updates remain within a reasonable bound, balancing exploration and stability.

3. **Entropy Coefficient (`ent_coef=0.05`)**:
   - Entropy encourages exploration by adding randomness to the policy. The entropy coefficient of `0.05` ensures that the model explores different actions rather than converging too quickly on suboptimal policies, making it more robust to varied situations.

4. **Discount Factor (`gamma=0.99`)**:
   - The discount factor determines how much the agent values future rewards compared to immediate rewards. A `gamma` value of `0.99` means that the agent considers long-term rewards almost as important as immediate ones, which is ideal for tasks like pose matching that require planning over several steps.

5. **Generalized Advantage Estimation (GAE) Lambda (`gae_lambda=0.95`)**:
   - This controls the bias-variance trade-off in advantage estimation. A GAE Lambda of `0.95` provides a good balance between accurate advantage estimation and stable updates.

6. **Batch Size (`batch_size=256`)**:
   - The batch size controls the number of samples processed before updating the model. A batch size of `256` strikes a balance between computational efficiency and learning stability, allowing for smoother gradient updates.

7. **Steps Per Environment Update (`n_steps=1024`)**:
   - This defines how many steps are taken in the environment before the model is updated. A large value like `1024` ensures that the model gathers sufficient experience before making updates, improving training stability.

8. **Maximum Gradient Norm (`max_grad_norm=0.5`)**:
   - This parameter clips the gradient to prevent excessively large updates. A `max_grad_norm` of `0.5` helps ensure that updates do not become too extreme, preventing instability in the learning process.

9. **Use State Dependent Exploration (`use_sde=True`)**:
   - State Dependent Exploration (SDE) is enabled to allow the agent to explore actions more efficiently. It adjusts exploration based on the state, making it more adaptable to the complexity of the environment.

### Phase 2: Dynamic Ground Truth Loading from JSON

In this phase, the reinforcement learning pipeline adapts to dynamically load ground truth quaternions from SMPL-X JSON files. This allows for flexibility in training the agent to replicate various hand poses, such as the ASL Letter "A".

#### Key Components:

1. **HandEnv Class**:
   - The environment is built using MuJoCo for physics-based simulation.
   - Ground truth quaternions are loaded from SMPL-X JSON files, allowing the agent to adapt to different target poses.
   - The action space is normalized between `-1` and `1` and scaled to control the robotic hand’s actuators in MuJoCo.

2. **Reward Calculation**:
   - Rewards are based on the similarity between the hand's rendered pose (in quaternion form) and the ground truth pose. A confidence score measures this similarity, with the goal of achieving a score above a defined threshold (95).

3. **RecurrentPPO Training**:
   - The agent is trained using RecurrentPPO, which incorporates LSTM for managing sequential data and learning temporal dependencies in actions.
   - Parallel environments (using DummyVecEnv) are used for faster training.
   - Hyperparameters include a learning rate of `0.0001`, a clip range of `0.4`, and steps per environment update (`1024`).

4. **Callback and Visualization**:
   - A custom callback logs the average rewards every 100 episodes, tracking the agent’s learning progress.
   - After training, the results are plotted (average reward vs episodes) and saved for analysis.

#### Dynamic Ground Truth Loading:
   - The ground truth quaternions are loaded dynamically from SMPL-X JSON files. These files provide rotation vectors for each joint, which are converted to quaternions and used as target poses for the agent to replicate.


