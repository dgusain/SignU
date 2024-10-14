import mujoco_py
import os
import time

# Load the model
xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

tendon_actuators = {
     "Wrist": [sim.model.actuator_name2id(name) for name in [
        "WRJ1r_motor", "WRJ1l_motor", "WRJ0u_motor", "WRJ0d_motor"]],
     "ForeFinger": [sim.model.actuator_name2id(name) for name in [
        "FFJ3r_motor", "FFJ3l_motor", "FFJ2u_motor", "FFJ2d_motor", "FFJ1u_motor", "FFJ1d_motor"]],
    "MiddleFinger": [sim.model.actuator_name2id(name) for name in [
        "MFJ3r_motor", "MFJ3l_motor", "MFJ2u_motor", "MFJ2d_motor", "MFJ1u_motor", "MFJ1d_motor"]],
    "RingFinger": [sim.model.actuator_name2id(name) for name in [
        "RFJ3r_motor", "RFJ3l_motor", "RFJ2u_motor", "RFJ2d_motor", "RFJ1u_motor", "RFJ1d_motor"]],
    "LittleFinger": [sim.model.actuator_name2id(name) for name in [
        "LFJ4u_motor", "LFJ4d_motor","LFJ3r_motor", "LFJ3l_motor", "LFJ2u_motor", "LFJ2d_motor", "LFJ1u_motor", "LFJ1d_motor"]],
    "Thumb": [sim.model.actuator_name2id(name) for name in [
        "THJ4a_motor", "THJ4c_motor","THJ3u_motor", "THJ3d_motor", "THJ2u_motor", "THJ2d_motor", "THJ1r_motor", "THJ1l_motor","THJ0r_motor", "THJ0l_motor"]],
}

# Function to set joint positions
def set_tendon_controls(tendon_ctrls, steps=200, delay=0.02):
    for i in range(steps):
        for tend, target_ctrl in tendon_ctrls.items():
            current_ctrl = sim.data.ctrl[tend]
            new_ctrl = current_ctrl + (target_ctrl - current_ctrl) * (i / steps)
            sim.data.ctrl[tend] = new_ctrl
        sim.step()
        viewer.render()
        time.sleep(delay)

asl_a_positions = {
    # Wrist
    tendon_actuators["Wrist"][0]: 0.0,  # WRJ1r_motor (pull)
    tendon_actuators["Wrist"][1]: 0.0,  # WRJ1l_motor (release)
    tendon_actuators["Wrist"][2]: 0.0,  # WRJ0u_motor (pull)
    tendon_actuators["Wrist"][3]: 0.0,  # WRJ0d_motor (release)

    # ForeFinger
    tendon_actuators["ForeFinger"][0]: 0.0,  # FFJ3r_motor (pull)
    tendon_actuators["ForeFinger"][1]: 0.0,  # FFJ3l_motor (release)
    tendon_actuators["ForeFinger"][2]: 0.0,  # FFJ2u_motor (pull)
    tendon_actuators["ForeFinger"][3]: -1.0,  # FFJ2d_motor (release)
    tendon_actuators["ForeFinger"][4]: 0.0,  # FFJ1u_motor (pull)
    tendon_actuators["ForeFinger"][5]: -1.0,  # FFJ1d_motor (release)

    # MiddleFinger
    tendon_actuators["MiddleFinger"][0]: 0.0,  # MFJ3r_motor (pull)
    tendon_actuators["MiddleFinger"][1]: 0.0,  # MFJ3l_motor (release)
    tendon_actuators["MiddleFinger"][2]: 0.0,  # MFJ2u_motor (pull)
    tendon_actuators["MiddleFinger"][3]: -1.0,  # MFJ2d_motor (release)
    tendon_actuators["MiddleFinger"][4]: 0.0,  # MFJ1u_motor (pull)
    tendon_actuators["MiddleFinger"][5]: -1.0,  # MFJ1d_motor (release)

    # RingFinger
    tendon_actuators["RingFinger"][0]: 0.0,  # RFJ3r_motor (pull)
    tendon_actuators["RingFinger"][1]: 0.0,  # RFJ3l_motor (release)
    tendon_actuators["RingFinger"][2]: 0.0,  # RFJ2u_motor (pull)
    tendon_actuators["RingFinger"][3]: -1.0,  # RFJ2d_motor (release)
    tendon_actuators["RingFinger"][4]: 0.0,  # RFJ1u_motor (pull)
    tendon_actuators["RingFinger"][5]: -1.0,  # RFJ1d_motor (release)

    # LittleFinger
    tendon_actuators["LittleFinger"][0]: 0.0,  # LFJ4u_motor (pull)
    tendon_actuators["LittleFinger"][1]: 0.0,  # LFJ4d_motor (release)
    tendon_actuators["LittleFinger"][2]: 0.0,  # LFJ3r_motor (pull)
    tendon_actuators["LittleFinger"][3]: 0.0,  # LFJ3l_motor (release)
    tendon_actuators["LittleFinger"][4]: 0.0,  # LFJ2u_motor (pull)
    tendon_actuators["LittleFinger"][5]: -1.0,  # LFJ2d_motor (release)
    tendon_actuators["LittleFinger"][6]: 0.0,  # LFJ1u_motor (pull)
    tendon_actuators["LittleFinger"][7]: -1.0,  # LFJ1d_motor (release)

    # Thumb
    tendon_actuators["Thumb"][0]: 0.0,  # THJ4a_motor (pull)
    tendon_actuators["Thumb"][1]: 0.0,  # THJ4c_motor (release)
    tendon_actuators["Thumb"][2]: -1.0,  # THJ3u_motor (pull)
    tendon_actuators["Thumb"][3]: 0.0,  # THJ3d_motor (release)
    tendon_actuators["Thumb"][4]: 0.0,  # THJ2u_motor (pull)
    tendon_actuators["Thumb"][5]: -1.0,  # THJ2d_motor (release)
    tendon_actuators["Thumb"][6]: 0.0,  # THJ1r_motor (pull)
    tendon_actuators["Thumb"][7]: -1.0,  # THJ1l_motor (release)
    tendon_actuators["Thumb"][8]: 0.0,  # THJ0r_motor (pull)
    tendon_actuators["Thumb"][9]: 0.0,  # THJ0l_motor (release)
}

asl_b_positions = {
    **{actuator: 0.0 for t in ["ForeFinger", "MiddleFinger", "RingFinger", "LittleFinger","Thumb","Wrist"] for actuator in tendon_actuators[t]},
    tendon_actuators["Thumb"][0]: 0.0,  # THJ4a_motor (pull)
    tendon_actuators["Thumb"][1]: -1.0,  # THJ4c_motor (release)
    tendon_actuators["Thumb"][2]: 0.0,  # THJ3u_motor (pull)
    tendon_actuators["Thumb"][3]: -1.0,  # THJ3d_motor (release)
    tendon_actuators["Thumb"][4]: 0.0,  # THJ2u_motor (pull)
    tendon_actuators["Thumb"][5]: -1.0,  # THJ2d_motor (release)
    tendon_actuators["Thumb"][6]: 0.0,  # THJ1r_motor (pull)
    tendon_actuators["Thumb"][7]: -1.0,  # THJ1l_motor (release)
    tendon_actuators["Thumb"][8]: 0.0,  # THJ0r_motor (pull)
    tendon_actuators["Thumb"][9]: 0.0,  # THJ0l_motor (release)
}

asl_revert = {
    **{actuator: 0.0 for t in ["ForeFinger", "MiddleFinger", "RingFinger", "LittleFinger","Thumb","Wrist"] for actuator in tendon_actuators[t]},
    #tendon_actuators["Wrist"][0]: -0.01,  # WRJ1r_motor (pull)
    #tendon_actuators["Wrist"][1]: 0.0,  # WRJ1l_motor (release)
    #tendon_actuators["Wrist"][2]: -0.01,  # WRJ0u_motor (pull)   
    #tendon_actuators["Wrist"][3]: 0.0,  # WRJ0d_motor (release)
    
    tendon_actuators["MiddleFinger"][2]: -1.0,  # FFJ2u_motor (pull)
    tendon_actuators["MiddleFinger"][4]: -1.0,  # MFJ1u_motor (pull)
    tendon_actuators["ForeFinger"][2]: -1.0,  # FFJ2d_motor (release)
    tendon_actuators["ForeFinger"][4]: -1.0,  # FFJ1d_motor (release)
    tendon_actuators["RingFinger"][2]: -1.0,  # RFJ2d_motor (release)
    tendon_actuators["RingFinger"][4]: -1.0,  # RFJ1d_motor (release)
    tendon_actuators["LittleFinger"][4]: -1.0,  # LFJ2d_motor (release)
    tendon_actuators["LittleFinger"][6]: -1.0,  # LFJ1d_motor (release)
    tendon_actuators["Thumb"][0]: -1.0,  # THJ3d_motor (release)
    tendon_actuators["Thumb"][2]: -1.0,  # THJ3d_motor (release)
    tendon_actuators["Thumb"][4]: -1.0,  # THJ2d_motor (release)
    tendon_actuators["Thumb"][6]: -1.0,  # THJ2d_motor (release)
}


wrist_adjust = {
    # Wrist
    tendon_actuators["Wrist"][0]: -0.05,  # WRJ1r_motor (pull)
    tendon_actuators["Wrist"][1]: 0.0,  # WRJ1l_motor (release)
    tendon_actuators["Wrist"][2]: -0.05,  # WRJ0u_motor (pull)
    tendon_actuators["Wrist"][3]: 0.0,  # WRJ0d_motor (release)
}

for i in range(3): 
	# Set the hand to display the letter 'A'
	set_tendon_controls(asl_a_positions)
	set_tendon_controls(wrist_adjust)
	time.sleep(0.02)  # Pause to view the letter 'A'
	set_tendon_controls(asl_revert)
	set_tendon_controls(asl_b_positions)
	set_tendon_controls(wrist_adjust)
	time.sleep(0.02)
	#set_tendon_controls(wrist_adjust)

# Keep the simulation running to view the result
while True:
    viewer.render()
    time.sleep(0.01)

