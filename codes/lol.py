from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os
import mujoco_py
import time
import cv2
from mujoco_py import MjRenderContextOffscreen
import base64
import requests

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

OLLAMA_MODEL = 'gpt-4o-mini'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model=OLLAMA_MODEL, temperature=0.0) 

actuator_value_template = """
You are an intelligent assistant who is learning to control a robotic hand. 
You can choose between the values given in the user input. 
Using your understanding along with the conversation history, provide a single value as output. 
If the value already exists in the conversation history, select another value. If all values have already been selected, return null.

Conversation History:
{conversation_history}

User: {user_input}
Value:
"""

act_prompt = PromptTemplate(
    input_variables=["conversation_history", "user_input"],
    template=actuator_value_template,
)
'''
response_template = """
You are an expert researcher with knowledge on American Sign Language and human hand gestures. Based on the user's input, tell if the image represents the letter 'A' in American Sign Language or not. 

Context:
{context}

User: {user_input}
Answer:
"""

response_prompt = PromptTemplate(
    input_variables=["context", "user_input"],
    template=response_template,
)
'''
# context in response_template will be the actuator position value selected by the llm. 

memory = ConversationBufferMemory(memory_key="conversation_history")
act_chain = LLMChain(
    llm=llm,
    prompt=act_prompt,
    verbose=False,
    memory=memory
)
'''
response_chain = LLMChain(
    llm=llm,
    prompt=response_prompt,
    verbose=False  # Removed memory=memory
)
'''
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# OpenAI API call
def call_openai_api_with_image(base64_image):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are an expert in American Sign language. What letter does this image represent? If it doesn't represent a letter, return 'irrelevant'."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 100
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

class Renderer:
    def __init__(self):
        xml_path = os.path.expanduser('/home/ducky/Downloads/Bot_hand/bot_hand.xml')
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.save_path = "rendered_img.jpg"
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
            
    def render_muj(self, action_value, image_path):
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
            self.finger_actuators["ForeFinger"][1]: action_value,
            self.finger_actuators["ForeFinger"][0]: action_value,
        })

        # Set the joint positions
        self.set_joint_positions(asl_a_positions)
        time.sleep(0.01)

        # Use offscreen rendering to capture the image
        offscreen = MjRenderContextOffscreen(self.sim, 0)
        offscreen.render(640, 480)  # Set desired resolution
        rendered_img = offscreen.read_pixels(640, 480, depth=False)
        img = cv2.rotate(rendered_img, cv2.ROTATE_180)
        img = cv2.flip(img, 1)
        save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, save_img)
        return 


def main():
    user_input = [0.0, 0.8, 1.6]
    trial = False
    fail_count = 3
    ind = 0
    while fail_count > 0:
        if trial:
            if done in user_input:
                user_input.remove(done)
        if len(user_input) == 0:
            break

        memory.save_context(
            {"user_input": f"values to choose from: {user_input}"}, 
            {"assistant": ""}
        )
        conversation_history = memory.load_memory_variables({})['conversation_history']
        act_value = act_chain.invoke({
            "conversation_history": conversation_history,
            "user_input": user_input
        })["text"].strip() 
        print("Actuator value: ", act_value)
        memory.save_context(
            {}, 
            {"assistant": f"Selected Value: {act_value}"}
        )
        done = float(act_value)
        # confirm that the act_value is within the user_input range
        if done not in user_input:
            fail_count -= 1   # three chances given to llm to generate an output within the input range
            continue
        else:
            trial = True
            ind += 1
        image_path = f"rendered_img{ind}.jpg"
        renderer = Renderer()
        renderer.render_muj(done, image_path)
        base64_image = encode_image(image_path)
        api_response = call_openai_api_with_image(base64_image)
        answer = api_response['choices'][0]['message']['content']
        print("Image Inference: ", answer)
        memory.save_context(
            {}, 
            {"assistant": f"Inference: {answer}"}
        )


if __name__ == "__main__":
    main()



