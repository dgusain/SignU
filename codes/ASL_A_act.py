import os
import mujoco_py
import time
import cv2
from mujoco_py import MjRenderContextOffscreen

class Renderer:
    def __init__(self):
        xml_path = os.path.expanduser('/home/dgusain/Bot_hand/bot_hand.xml')
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        #self.viewer = mujoco_py.MjViewer(self.sim)
        
        self.finger_actuators = {
            "ForeFinger": [self.sim.model.actuator_name2id(name) for name in [
                "ForeFingerJoint0_act", "ForeFingerJoint1_act", "ForeFingerJoint2_act", "ForeFingerJoint3_act"]],
            "MiddleFinger": [self.sim.model.actuator_name2id(name) for name in [
                "MiddleFingerJoint0_act", "MiddleFingerJoint1_act", "MiddleFingerJoint2_act", "MiddleFingerJoint3_act"]],
            "RingFinger": [self.sim.model.actuator_name2id(name) for name in [
                "RingFingerJoint0_act", "RingFingerJoint1_act", "RingFingerJoint2_act", "RingFingerJoint3_act"]],
            "LittleFinger": [self.sim.model.actuator_name2id(name) for name in [
                "LittleFingerJoint0_act", "LittleFingerJoint1_act", "LittleFingerJoint2_act", "LittleFingerJoint3_act"]],
            "Thumb": [self.sim.model.actuator_name2id(name) for name in [
                "ThumbJoint0_act", "ThumbJoint1_act", "ThumbJoint2_act", "ThumbJoint3_act", "ThumbJoint4_act"]],
        }

    def set_joint_positions(self, joint_positions, steps=200, delay=0.02):
        for i in range(steps):
            for joint, target_position in joint_positions.items():
                current_position = self.sim.data.ctrl[joint]
                new_position = current_position + (target_position - current_position) * (i / steps)
                self.sim.data.ctrl[joint] = new_position
            self.sim.step()
            #self.viewer.render()
            time.sleep(delay)
            
    def render_muj(self, action_value):
        asl_a_positions = {
            actuator: 1.6 for finger in ["ForeFinger", "MiddleFinger", "RingFinger", "LittleFinger"] 
            for actuator in self.finger_actuators[finger]
        }
        asl_a_positions.update({
            self.finger_actuators["ForeFinger"][3]: -0.01,
            self.finger_actuators["MiddleFinger"][3]: -0.01,
            self.finger_actuators["RingFinger"][3]: -0.01,
            self.finger_actuators["LittleFinger"][3]: -0.01,
            self.finger_actuators["Thumb"][0]: -0.9,
            self.finger_actuators["Thumb"][1]: 0.0,
            self.finger_actuators["Thumb"][2]: 0.262,
            self.finger_actuators["Thumb"][3]: 0.5,
            self.finger_actuators["Thumb"][4]: 0.5,
            self.finger_actuators["ForeFinger"][2]: action_value,
        })

        # Set the joint positions
        self.set_joint_positions(asl_a_positions)

        # Use offscreen rendering to capture the image
        #os.environ["MUJOCO_GL"] = "egl"
        offscreen = MjRenderContextOffscreen(self.sim, 0)
        offscreen.render(640, 480)  # Set desired resolution
        rendered_img = offscreen.read_pixels(640, 480, depth=False)

        return rendered_img


def main():
    user_input = 1.6
    renderer = Renderer()
    img = renderer.render_muj(user_input)
    img = cv2.rotate(img, cv2.ROTATE_180)
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imshow("Rendered Pose", img)
    cv2.imwrite("rendered_pose.png", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
