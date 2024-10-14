import mujoco_py
import os
import time

# Load the model
xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Actuator indices for the finger and thumb joints based on the uploaded actuators_for_joints.xml
finger_actuators = {
    "ForeFinger": [sim.model.actuator_name2id(name) for name in [
        "ForeFingerJoint0_act", "ForeFingerJoint1_act", "ForeFingerJoint2_act", "ForeFingerJoint3_act"]],
    "MiddleFinger": [sim.model.actuator_name2id(name) for name in [
        "MiddleFingerJoint0_act", "MiddleFingerJoint1_act", "MiddleFingerJoint2_act", "MiddleFingerJoint3_act"]],
    "RingFinger": [sim.model.actuator_name2id(name) for name in [
        "RingFingerJoint0_act", "RingFingerJoint1_act", "RingFingerJoint2_act", "RingFingerJoint3_act"]],
    "LittleFinger": [sim.model.actuator_name2id(name) for name in [
        "LittleFingerJoint0_act", "LittleFingerJoint1_act", "LittleFingerJoint2_act", "LittleFingerJoint3_act"]],
    "Thumb": [sim.model.actuator_name2id(name) for name in [
        "ThumbJoint0_act", "ThumbJoint1_act", "ThumbJoint2_act", "ThumbJoint3_act", "ThumbJoint4_act"]],
}


wrist_actuators = {
	"Wrist": [sim.model.actuator_name2id(name) for name in [
        "WristJoint0_act", "WristJoint1_act"]],
}



def set_joint_positions(joint_positions, steps=200, delay=0.02):
    for i in range(steps):
        for joint, target_position in joint_positions.items():
            current_position = sim.data.ctrl[joint]
            new_position = current_position + (target_position - current_position) * (i / steps)
            sim.data.ctrl[joint] = new_position
        sim.step()
        viewer.render()
        time.sleep(delay)
        
# Joint positions for wrist
asl_z0_positions = {
    wrist_actuators["Wrist"][1]: -0.524,  # WristJoint0_act (wrist) [-0.524 0.175]
    wrist_actuators["Wrist"][0]: -0.785,  # WristJoint0_act (palm) [-0.785 0.611]
    **{actuator: 1.6 for finger in ["MiddleFinger", "RingFinger", "LittleFinger"] for actuator in finger_actuators[finger]},
}

asl_z1_positions = {
    wrist_actuators["Wrist"][1]: 0.175,  # WristJoint0_act (wrist) [-0.524 0.175]
    wrist_actuators["Wrist"][0]: -0.785,  # WristJoint0_act (palm) [-0.785 0.611]

}

asl_z2_positions = {
    wrist_actuators["Wrist"][1]: 0.175,  # WristJoint0_act (wrist) [-0.524 0.175]
    wrist_actuators["Wrist"][0]: 0.611,  # WristJoint0_act (palm) [-0.785 0.611]

}

asl_z3_positions = {
    wrist_actuators["Wrist"][1]: -0.524,  # WristJoint0_act (wrist) [-0.524 0.175]
    wrist_actuators["Wrist"][0]: 0.611,  # WristJoint0_act (palm) [-0.785 0.611]

}
# THINK OF IT AS A CIRCLE OR ELLIPSE. 2D COORDINATE SYSTEM. 4 POINTS. 
# DISTANCE FROM Z0 TO Z1: 0.175+0.524 = 0.699  -- MINOR AXIS
# DISTANCE FROM Z1 TO Z2: 0.611 + 0.785 = 1.396  -- MAJOR AXIS
# TO MAKE THE WRIST SWING, WE MAKE IT TRACE THE CIRCUMFERENCE OF THE ELLIPSE. 

asl_revert = { **{actuator: 0.0 for w in ["Wrist"] for actuator in wrist_actuators[w]}, 
}


for i in range(5): 
	set_joint_positions(asl_z0_positions,200,0.01)
	set_joint_positions(asl_z1_positions,200,0.01)
	set_joint_positions(asl_z2_positions,200,0.01)
	set_joint_positions(asl_z3_positions,200,0.01)
	time.sleep(0.02)  
	
	#set_joint_positions(asl_revert,100, 0.01)

# Keep the simulation running to view the result
while True:
    viewer.render()
    time.sleep(0.01)



