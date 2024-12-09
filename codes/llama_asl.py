import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor  
import os
import time
from typing import List, Optional, Dict
import io
import requests

model_id = "meta-llama/Llama-3.2-11B-Vision"
MAIN_CACHE_DIR: str = "/home/dgusain/Bot_hand/Huggingface/"  
MODEL_NAME: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LOCAL_MODEL_DIR: str = os.path.join(MAIN_CACHE_DIR, MODEL_NAME)  
HUGGINGFACE_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
device = torch.device("cuda:0")
model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        cache_dir=MAIN_CACHE_DIR,
        use_auth_token=HUGGINGFACE_TOKEN,  
        torch_dtype=torch.float16,  
).half().to(device)
processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=MAIN_CACHE_DIR,
        use_auth_token=HUGGINGFACE_TOKEN,  
)

if __name__ == "__main__":
    save_path = "two.jpeg"
    image = Image.open(save_path).convert("RGB")
    prompt = (
        "<|image|><|begin_of_text|> You are an expert in American Sign Language (ASL) and human hand gestures. "
        "What letter in ASL does it correspond to? "
        "If it does not correspond to a letter, inform the user."
     "Answer: "
    )
    #prompt = "<|image|><|begin_of_text|> You are an expert with knowledge about American Sign Language and human hand gestures. What can you infer about the gesture given in this picture?"
    inputs = processor(image, prompt, return_tensors="pt", padding=True,truncation=True, max_length=4096).to(model.device)
    output = model.generate(**inputs, max_new_tokens=150)
    decoded_output = processor.decode(output[0], skip_special_tokens=True)
    print(decoded_output)

# the response is not correct
# maybe it needs more images to understand what i am trying to say

