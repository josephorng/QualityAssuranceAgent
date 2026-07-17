import base64
from pathlib import Path

import requests

# 1. Base64 encode the image
image_path = Path(__file__).resolve().parent / "test_images" / "wm7_0_0001.jpg"
with open(image_path, "rb") as image_file:
    b64_image = base64.b64encode(image_file.read()).decode("utf-8")

# 2. Construct the message with content as an array of objects
payload = {
    "model": "google/gemma-4-26B-A4B-it",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Describe this image carefully."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 512,
    "temperature": 0.0
}

# 3. Post to the vLLM server
response = requests.post(
    "http://192.168.4.134:8000/v1/chat/completions", 
    json=payload
)
print(response.json()["choices"][0]["message"]["content"])