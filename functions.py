import requests
import os
from openai import OpenAI
from .constants import HF_BASE_LINK

def api_free(prompt):
    url = "https://apifreellm.com/api/chat"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "message": f"{prompt}"
    }
    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    if result.get("status") == "success":
        return result["response"]
    else:
        "Error:", result["error"]

def api_huggingface(prompt=None, messages=None):
    client = OpenAI(
    base_url=HF_BASE_LINK,
    api_key=os.environ['HF_TOKEN']
    )

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[
            {
                "role": "user",
                "content": f"{prompt}"
            }
        ] if prompt else messages
    )

    return completion.choices[0].message

def api_local(prompt, model):
    completion = model(f"{prompt}")
    return completion[0]['generated_text']