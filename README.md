
# Project: Reinforcement Learning for American Sign Language Fingerspelling in MuJoCo

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

Leveraged Proximal Policy Algorithm with recurrent neural networks, with custom Actor-Critic (A2C) policy, to learn a generalized, optimal policy to generate ASL hand gestures. 


