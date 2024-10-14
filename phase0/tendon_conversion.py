# SMPLX joint data for right hand (as axis-angle) provided by user
right_hand_pose_axis_angle = [
    [-0.1512831449508667, -0.15869110822677612, 0.8509026765823364],  # Example for Thumb MCP
    [0.3627489507198334, 0.022802263498306274, 0.877955436706543],     # Thumb Proximal
    [-0.25605103373527527, 0.23274479806423187, -0.12487145513296127], # Thumb Distal
    [-0.3255625367164612, -0.011992461048066616, 1.1573609113693237],  # Index MCP
    [0.026095140725374222, -0.046903207898139954, 0.3613745868206024], # Index PIP
    [0.007650915998965502, 0.07361442595720291, 0.3832685649394989],   # Index DIP
    [-0.4402444660663605, 0.40829408168792725, 0.5904672145843506],    # Middle MCP
    [-0.3253646492958069, 0.07266436517238617, 0.3685110807418823],    # Middle PIP
    [-0.18729417026042938, 0.09435857087373734, 0.4643711745738983],   # Middle DIP
    [-0.15461142361164093, 0.29494309425354004, 0.9347426891326904],   # Ring MCP
    [-0.30943793058395386, 0.19789335131645203, 0.34914878010749817],  # Ring PIP
    [-0.14297737181186676, -0.1090855672955513, 0.4367266297340393],   # Ring DIP
    [0.07962226122617722, -0.15232424437999725, -0.13116824626922607], # Little MCP
    [-0.1877722293138504, 0.2988579273223877, 0.05618938431143761],    # Little PIP
    [0.03166203200817108, 0.09782802313566208, -0.1685519814491272]    # Little DIP
]

# Define maximum flexion angle (assumed to be 90 degrees or π/2 radians)
max_flexion_angle = np.pi / 2

# Function to convert axis-angle to angular displacement
def axis_angle_to_angular_displacement(axis_angle):
    """Convert axis-angle to the angle of rotation (angular displacement)."""
    return np.linalg.norm(axis_angle)

# Function to map angular displacement to control input
def map_angle_to_control(angle, max_angle=max_flexion_angle):
    """Map angular displacement to control input range [-1, 0]."""
    control_input = -angle / max_angle
    return np.clip(control_input, -1, 0)

# Calculate control inputs for all tendons
control_inputs = []
for axis_angle in right_hand_pose_axis_angle:
    angular_displacement = axis_angle_to_angular_displacement(axis_angle)  # Get the rotation magnitude
    control_input = map_angle_to_control(angular_displacement)  # Map to tendon control input
    control_inputs.append(control_input)

# Output control inputs for tendons
control_inputs
