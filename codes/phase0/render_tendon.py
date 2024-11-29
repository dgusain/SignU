import mujoco_py
import os
import time
import numpy as np

# Path to your bot_hand.xml file
xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')

# Load the MuJoCo model
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Function to retrieve actuator indices safely
def get_actuator_id(actuator_name, model):
    try:
        return model.actuator_name2id(actuator_name)
    except ValueError:
        print(f"Actuator '{actuator_name}' not found in the model.")
        exit(1)

# Retrieve actuator IDs for the Forefinger MCP Joint Tendons
# Assuming 'ForeFingerJoint2' is the MCP Joint for the Forefinger
# From your XML, 'FFJ2u_motor' and 'FFJ2d_motor' influence 'ForeFingerJoint2'
FFJ2u_motor = 'FFJ2u_motor'  # Tendon actuator to flex the MCP joint
FFJ2d_motor = 'FFJ2d_motor'  # Tendon actuator to extend the MCP joint (if needed)


ffj2u_id = get_actuator_id(FFJ2u_motor, model)
ffj2d_id = get_actuator_id(FFJ2d_motor, model)

# Initialize all control inputs to zero
sim.data.ctrl[:] = 0.0

# Define the control values
# Since ctrlrange for tendons is typically [-1, 0], where -1 is full activation
# and 0 is no activation, setting FFJ2u_motor to -1 will attempt to flex the MCP joint
# and FFJ2d_motor to 0 ensures no opposing force is applied
sim.data.ctrl[ffj2u_id] = -1.0  # Max flexion
sim.data.ctrl[ffj2d_id] = 0.0   # No extension

# Optional: Print actuator indices and names for verification
print(f"Actuator '{FFJ2u_motor}' has index {ffj2u_id}. Setting to -1 (max flexion).")
print(f"Actuator '{FFJ2d_motor}' has index {ffj2d_id}. Setting to 0 (no extension).")

# Simulation parameters
simulation_time = 5.0  # seconds to apply the control
start_time = time.time()

# Main simulation loop
while True:
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Apply control for the specified duration
    if elapsed_time < simulation_time:
        # Ensure the control signals remain constant
        sim.data.ctrl[ffj2u_id] = -1.0
        sim.data.ctrl[ffj2d_id] = 0.0
    else:
        # After the duration, reset control signals to zero
        sim.data.ctrl[ffj2u_id] = 0.0
        sim.data.ctrl[ffj2d_id] = 0.0

    # Step the simulation
    sim.step()
    viewer.render()

    # Optional: Exit after some time to prevent infinite loop
    if elapsed_time > (simulation_time + 2.0):  # Additional time to observe reset
        print("Control test completed. Exiting simulation.")
        break

# Keep the viewer open until manually closed
while True:
    sim.step()
    viewer.render()

