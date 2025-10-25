from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    token=os.getenv("HF_TOKEN")
)

# New-style chat/completion API
response = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=[
        {"role": "user", "content": "Blood is coming while fapping?"}
    ],
)

# print the model's reply
print(response.choices[0].message["content"])
