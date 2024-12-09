import openai

# Set up OpenAI API
openai.api_key = "your-openai-api-key"

def query_gpt4(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Example use case: Reward function adjustment
prompt = "The reward function penalizes the agent heavily when joint angles deviate. Suggest improvements."
print(query_gpt4(prompt))
