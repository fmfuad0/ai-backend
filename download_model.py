import os
import requests

MODEL_PATH = "tinyllama.gguf"

if not os.path.exists(MODEL_PATH):
    print("Downloading TinyLlama model...")
    url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    with requests.get(url, stream=True) as r:
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Model downloaded successfully!")
else:
    print("Model already exists")
