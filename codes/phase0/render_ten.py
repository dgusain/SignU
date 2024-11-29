import mujoco_py
import os
import time
import numpy as np

# Path to your bot_hand.xml file
xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')

# Check if the XML file exists
if not os.path.exists(xml_path):
    print(f"Error: The XML file '{xml_path}' does not exist.")
    exit(1)

# Load the MuJoCo model
try:
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)

# Function to retrieve actuator IDs for tendon-based actuators
def get_tendon_actuators(model):
    tendon_actuators = []
    for idx, name in enumerate(model.actuator_names):
        tendon_actuators.append((name, idx))
    return tendon_actuators

# Retrieve all tendon-based actuators
tendon_actuators = get_tendon_actuators(model)

if not tendon_actuators:
    print("No tendon-based actuators found in the model.")
    exit(1)

# Display tendon actuators
print("Tendon-Based Actuators:")
print("------------------------")
for name, idx in tendon_actuators:
    print(f"Actuator Name: {name}, Actuator ID: {idx}")
print("------------------------\n")

# Set tendon actuators to maintain a stationary pose
def maintain_stationary_pose(sim, model):
    # Set control values for specific tendons to maintain pose
    control_values = {
        'FFJ1u_motor': -0.5,  # Example values, adjust based on your specific needs
        'FFJ2u_motor': -0.5,
        'MFJ1u_motor': -0.5,
        'RFJ1u_motor': -0.5,
        'LFJ1u_motor': -0.5,
        'THJ1u_motor': -0.5, 
        'WRJ1u_motor': -0.5, 
        'WRJ0u_motor': -0.5,  # Thumb motor, adjust for upward pose
        # Add more tendons as needed
    }

    for actuator_name, value in control_values.items():
        set_tendon_actuator(sim, model, actuator_name, value)

def set_wrist_pose(sim):
    # Lock wrist and forearm in specific positions
    wrist_pose = [0, 0, np.pi / 2]  # Assuming this keeps the palm facing outward
    sim.data.qpos[:len(wrist_pose)] = wrist_pose
    
# Function to set a specific tendon actuator
def set_tendon_actuator(sim, model, actuator_name, value):
    try:
        actuator_id = model.actuator_name2id(actuator_name)
    except ValueError:
        print(f"Actuator '{actuator_name}' not found in the model.")
        return

    # Retrieve the control range for the actuator
    ctrl_min = model.actuator_ctrlrange[actuator_id][0]
    ctrl_max = model.actuator_ctrlrange[actuator_id][1]

    # Clamp the value within the control range
    clamped_value = max(min(value, ctrl_max), ctrl_min)

    # Set the control value
    sim.data.ctrl[actuator_id] = clamped_value
    print(f"Set '{actuator_name}' (ID: {actuator_id}) to {clamped_value}.")

# Gravity compensation: Apply a force equal and opposite to gravity
def set_gravity_compensation(sim, model):
    gravity = sim.model.opt.gravity.copy()
    for i in range(len(sim.data.qfrc_applied)):
        sim.data.qfrc_applied[i] = -sim.data.qfrc_bias[i]

# Apply gravity compensation and maintain the pose
def apply_control(sim, model):
    set_gravity_compensation(sim, model)
    maintain_stationary_pose(sim, model)

# Simulation parameters
simulation_time = 5.0  # Duration to apply the control in seconds
start_time = time.time()

print("Starting simulation. Maintaining stationary hand pose with gravity compensation applied.\n")

# Run the simulation
try:
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Apply gravity compensation and control continuously
        apply_control(sim, model)

        # Step the simulation
        sim.step()
        viewer.render()

        # Optional: Reset simulation after a certain time
        #if elapsed_time > simulation_time:
         #   print("Simulation time reached. Resetting tendon actuators to rest.")
          #  for name, idx in tendon_actuators:
           #     sim.data.ctrl[idx] = 0.0
            #break

        # Sleep to sync with real-time (optional)
        time.sleep(sim.model.opt.timestep)
except KeyboardInterrupt:
    print("\nSimulation interrupted by user. Exiting...")

# Keep the viewer open until manually closed
while True:
    sim.step()
    viewer.render()

