import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco_py
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from multiprocessing import Process

class HandEnv(gym.Env):
    def __init__(self):
        super(HandEnv, self).__init__()
        xml_path = os.path.expanduser('/home/easgrad/dgusain/Bot_hand/bot_hand.xml')
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)

        # Initialize min and max actuator ranges
        actuator_min = []
        actuator_max = []

        self.finger_actuators = {
            "ForeFinger": [self.sim.model.actuator_name2id(name) for name in [
                "ForeFingerJoint0_act", "ForeFingerJoint1_act", "ForeFingerJoint2_act", "ForeFingerJoint3_act"]],
            "MiddleFinger": [self.sim.model.actuator_name2id(name) for name in [
                "MiddleFingerJoint0_act", "MiddleFingerJoint1_act", "MiddleFingerJoint2_act", "MiddleFingerJoint3_act"]],
            "RingFinger": [self.sim.model.actuator_name2id(name) for name in [
                "RingFingerJoint0_act", "RingFingerJoint1_act", "RingFingerJoint2_act", "RingFingerJoint3_act"]],
            "LittleFinger": [self.sim.model.actuator_name2id(name) for name in [
                "LittleFingerJoint0_act", "LittleFingerJoint1_act", "LittleFingerJoint2_act", "LittleFingerJoint3_act","LittleFingerJoint4_act"]],
            "Thumb": [self.sim.model.actuator_name2id(name) for name in [
                "ThumbJoint0_act", "ThumbJoint1_act", "ThumbJoint2_act", "ThumbJoint3_act", "ThumbJoint4_act"]],
        }
        self.wrist_actuators = {
            "Wrist": [self.sim.model.actuator_name2id(name) for name in [
                "WristJoint0_act", "WristJoint1_act"]],
        }

        # Collect actuator min and max values for each actuator
        for finger, actuators in self.finger_actuators.items():
            for actuator_id in actuators:
                actuator_range = self.sim.model.actuator_ctrlrange[actuator_id]
                actuator_min.append(actuator_range[0])
                actuator_max.append(actuator_range[1])

        for w, wr_actuators in self.wrist_actuators.items():
            for actuator_id in wr_actuators:
                actuator_range = self.sim.model.actuator_ctrlrange[actuator_id]
                actuator_min.append(actuator_range[0])
                actuator_max.append(actuator_range[1])

        # Convert min and max values to numpy arrays for easier manipulation
        self.actuator_min = np.array(actuator_min)
        self.actuator_max = np.array(actuator_max)

        # Create normalized action space between [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.actuator_min),), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        return self.sim.data.qpos[:24].astype(np.float32), {}

    def step(self, action):
        # Rescale action from [-1, 1] to the actuator's actual range
        rescaled_action = self.actuator_min + (action + 1) * (self.actuator_max - self.actuator_min) / 2

        self.sim.data.ctrl[:] = rescaled_action
        self.sim.step()

        state = self.sim.data.qpos[:24].astype(np.float32)
        reward = float(self.calculate_reward(state))
        done = reward >= 50
        truncated = False
        info = {}
        if done:
            pass  # Implement logging or other actions if necessary

        return state, reward, done, truncated, info

    def calculate_reward(self, state):
        ground_truth_quat = self.get_ground_truth_quaternion()
        rendered_quat = self.get_rendered_pose_quaternion()

        # Normalize quaternions and compute per-joint confidence score
        confidence_scores = []
        for gt_q, r_q in zip(ground_truth_quat, rendered_quat):
            # Normalize individual quaternions
            gt_q_norm = self.normalize_quaternion(gt_q)
            r_q_norm = self.normalize_quaternion(r_q)
            confidence_score = self.compute_quaternion_similarity(r_q_norm, gt_q_norm)
            confidence_scores.append(confidence_score)

        # Average confidence scores across joints
        avg_confidence_score = np.mean(confidence_scores)
        
        return avg_confidence_score - 50  # Reward calculation

    def get_rendered_pose_quaternion(self):
        quaternions = []
        for joint in range(self.model.njnt):  # Iterate over all joints in the model
            quaternion = self.sim.data.qpos[self.model.jnt_qposadr[joint]:self.model.jnt_qposadr[joint] + 4]
            quaternions.append(quaternion)
        
        return np.array(quaternions)  # Return as a numpy array

    def compute_quaternion_similarity(self, q1, q2):
        """Compute the dot product for quaternion similarity."""
        dp = np.dot(q1.flatten(), q2.flatten())
        similarity = dp * 100
        return similarity

    def normalize_quaternion(self, q):
        """Normalize each quaternion to ensure it's a unit quaternion."""
        norm = np.linalg.norm(q)
        return q / norm if norm != 0 else q  # Prevent division by zero

    def get_ground_truth_quaternion(self):
        """
        Returns the ground truth quaternions for all joints as a numpy array.
        The provided quaternions will be used for comparison with the rendered pose.
        """
        ground_truth_quats = [
            [ 6.32658406e-04,  1.25473697e-03, -9.99718591e-03,  1.56702220e+00],
            [ 1.25473697e-03, -9.99718591e-03,  1.56702220e+00,  1.56730258e+00],
            [-0.00999719,  1.5670222,   1.56730258,  1.5674143 ],
            [ 1.5670222,   1.56730258,  1.5674143,  -0.00999801],
            [ 1.56730258,  1.5674143,  -0.00999801,  1.56701926],
            [ 1.5674143,  -0.00999801,  1.56701926,  1.56730194],
            [-0.00999801,  1.56701926,  1.56730194,  1.5674143 ],
            [ 1.56701926,  1.56730194,  1.5674143,  -0.00999756],
            [ 1.56730194,  1.5674143,  -0.00999756,  1.56701717],
            [ 1.5674143,  -0.00999756,  1.56701717,  1.56730177],
            [-0.00999756,  1.56701717,  1.56730177,  1.56741429],
            [1.56701717, 1.56730177, 1.56741429, 0.00868252],
            [ 1.56730177,  1.56741429,  0.00868252, -0.01000805],
            [ 1.56741429,  0.00868252, -0.01000805,  1.56708349],
            [ 0.00868252, -0.01000805,  1.56708349,  1.56730911],
            [-0.01000805,  1.56708349,  1.56730911,  1.56741508],
            [1.56708349, 1.56730911, 1.56741508, 0.49916711],
            [1.56730911, 1.56741508, 0.49916711, 0.50022545],
            [1.56741508, 0.49916711, 0.50022545, 0.25330455],
            [4.99167109e-01, 5.00225452e-01, 2.53304550e-01, 4.08440608e-04],
            [ 5.00225452e-01,  2.53304550e-01,  4.08440608e-04, -9.00000689e-01],
            [ 2.53304550e-01,  4.08440608e-04, -9.00000689e-01,  1.00000000e-01],
            [ 4.08440608e-04, -9.00000689e-01,  1.00000000e-01, -1.00000000e-01],
            [-0.90000069,  0.1,        -0.1,         0.01463168],
            [ 0.1,        -0.1,         0.01463168,  1.        ]
        ]

        return np.array(ground_truth_quats)  # Return as numpy array for comparison

def train_on_gpu(gpu_id, iterations_per_gpu):
    """
    Function to train a PPO model on a specified GPU.

    Args:
        gpu_id (int): The GPU ID to use for this training process.
        iterations_per_gpu (int): The number of training iterations for this GPU.
    """
    # Assign the specific GPU to this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"GPU {gpu_id}: Using {device} device")

    # Monitoring and Logging Setup
    #log_dir = f"./logs/gpu_{gpu_id}/"
    #os.makedirs(log_dir, exist_ok=True)
    env = HandEnv()
    
    # Initialize PPO model with the specified device
    model = PPO('MlpPolicy',env,verbose=1,device=device,ent_coef=0.01,learning_rate=0.0001,clip_range=0.3,n_steps=4096)
    
    # Custom training loop with logging
    for iteration in range(iterations_per_gpu):
        model.learn(total_timesteps=1024, reset_num_timesteps=False)
    
        # After each iteration, evaluate the model and log the reward
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(1000):  # Specify the number of steps for evaluation
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done:
                break
        print(f"GPU {gpu_id} - Iteration {iteration + 1}/{iterations_per_gpu} - Reward: {reward}")
    
    # Save the model uniquely per GPU
    model.save(f"hand_pose_ppo_gpu_{gpu_id}")

def main():
    """
    Main function to initiate training on multiple GPUs.
    """
    num_gpus = 7  # GPUs with IDs 0 to 6
    total_iterations = 700  # Total iterations you want across all GPUs
    iterations_per_gpu = total_iterations // num_gpus
    remaining_iterations = total_iterations % num_gpus

    processes = []
    
    for gpu_id in range(num_gpus):
        # Distribute the remaining iterations among the first few GPUs
        iters = iterations_per_gpu + (1 if gpu_id < remaining_iterations else 0)
        p = Process(target=train_on_gpu, args=(gpu_id, iters))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
