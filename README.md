# SignMimic:
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

"""
   
