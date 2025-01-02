
# SignU: Training a MuJoCo-based robotic hand to perform American Sign Language gestures

## Context
This project leverages reinforcement learning to teach an agent to form various hand gestures for fingerspelling in American Sign Language (ASL). The agent learns to form letters like "A", "B", etc., autonomously. Secondly, this project aims to develop the world's first mapping strategy, linking SMPLX 3D body models to a reality based phyics engine like MuJoCo. 

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

### Objectives:
- **Phase 1**: Reinforcement learning of all American Sign Language letters by a MuJoCo model
- **Phase 2**: Mapping SMPLX 3D body model data to a MuJoCo model

### Progress:
1. **Manual movement**: Successfully performed manual maneuvers of the MuJoCo Shadow Dexterous Hand model, forming all 26 letters of ASL. 
2. **Reinforcement pipeline**: Designed a RL pipeline capable of learning optimal policy to form 16 gestures on its own. 
3. **Anatomical Mapping**: Created an anatomical mapping strategy, to map 15 SMPLX quaternion joint data to 24 MuJoCo quaternion joint data.
4. **Generalized Strategy**: Generalizing the anatomical mapping to all 26 ASL letters.


## Phase 1: RL pipeline
The approach integrates a custom environment design, a deep neural network architecture with residual blocks, and implementation of Proximal Policy Optimization (PPO) algorithm enhanced with a recurrent layers.  
### Environment Setup: 
The reinforcement learning agent operates within a custom-made Gym environment designed to simulate the dynamics of a robotic hand using the MuJoCo physics engine. The environment, defined in a ‘HandEnv‘ class, includes:
 - MuJoCo Model Loading: The environment loads a MuJoCo XML model (Shadow Dexterous Hand) specifying the hand’s physical properties, joint actuator identifiers and control ranges.
 - Observation Space: A continuous box space of shape (96,) representing 24 joints, each described by a 4-dimensional quaternion.
 - Action Space: A multi-discrete space where each actuator can select from 11 discrete actions (0 to 10) corresponding to target positions within its control range. Based on previous iterations, a discretized action space shows better performance as compared to a continuous action space.
 - Reward Mechanism: The reward is based on the angular difference between the quaternions obtained from the agent’s current rendered pose and the desired pose, encouraging accurate joint positioning.
 - Episode Configuration: Each episode consists of a single step with 200 internal rendering steps, followed by an environment reset, fostering precise control at the per-step level.
### Neural network architecture: 
A custom neural network architecture is designed to process observations and produce actions. The architecture integrates residual blocks to facilitate deeper representation
learning and employs an LSTM layer to capture temporal dependencies. The network includes both policy and value networks.
| **Layer**                  | **Type**       | **Output Size**        | **Activation** |
|-----------------------------|---------------|------------------------|---------------|
| **Input**                  | -             | 96                     | -             |
| **Initial Fully Connected**| Linear        | 256                    | ReLU          |
| **Initial Layer Normalization** | LayerNorm   | 256                    | -             |
| **Residual Blocks**        |               |                        |               |
|                             | Residual Block 1 | 256                    | ReLU          |
|                             | Residual Block 2 | 256                    | ReLU          |
|                             | Residual Block 3 | 256                    | ReLU          |
|                             | Residual Block 4 | 256                    | ReLU          |
|                             | Residual Block 5 | 256                    | ReLU          |
| **Final Fully Connected**  | Linear        | 2048                   | ReLU          |
| **Final Layer Normalization** | LayerNorm   | 2048                   | -             |
| **Policy Network MLP**     | MLP           | (2048, 2048, 1024)     | ReLU          |
| **Value Network MLP**      | MLP           | (2048, 2048, 1024)     | ReLU          |
| **LSTM Layer**             | LSTM          | 1024                   | -             |

#### Residual blocks: 
Each residual block consists of two linear layers with ReLU activations and dropout for
regularization. A residual connection bypasses these layers, aiding in gradient flow and
mitigating vanishing gradients.
#### Custom Feature Extractor:
 - Initial Fully Connected Layer: Maps 96-dimensional input to 256 dimensions, followed by ReLU and layer normalization.
 - Five Residual Blocks: Each block refines the representation, maintaining stable gradient flow across 256 input-output dimensions.
 - Final Fully Connected Layer: Projects features to 2048 dimensions with ReLU and layer normalization.
#### Actor-Critic Custom LSTM Policy:
Designed a custom Actor-Critic (A2C) policy with LSTM layers to extend over the initial Multi-layer perceptron policy, in order to integrate a custom Feature Extractor. The policy and value networks have three fully connected layers, each with architectures (2048, 2048, 1024). A LSTM layer with a hidden size of 1024 captures temporal dependencies.

### Results: 
<table>
<tr>
     <img src="https://github.com/dgusain/SignU/blob/main/gallery/results_AN.png" alt="SignMimic" width="600" height="600">  
</tr>
<tr>
     <img src="https://github.com/dgusain/SignU/blob/main/gallery/results_VW.png" alt="SignMimic" width="600" height="600">  
</tr>
</table>



