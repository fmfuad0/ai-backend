from fastapi import FastAPI
from pydantic import BaseModel
from ctransformers import AutoModelForCausalLM
import os
import requests

app = FastAPI()

MODEL_PATH = "tinyllama.gguf"

# ✅ Auto-download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    with requests.get(url, stream=True) as r:
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Model downloaded.")

# ✅ Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    ".", model_file=MODEL_PATH, model_type="llama"
)
print("Model loaded!")

# ------------------ Request Models ------------------
class EditRequest(BaseModel):
    code: str
    instruction: str

class ChatRequest(BaseModel):
    message: str

# ------------------ Endpoints ------------------
@app.get("/")
def root():
    return {"status": "running"}

@app.post("/edit")
def edit(req: EditRequest):
    """
    AI Code Edit Endpoint
    """
    prompt = f"""
You are a helpful coding assistant.

Instruction: {req.instruction}

Code:
{req.code}
"""
    output = model(prompt, max_new_tokens=150)
    return {"edited_code": output}

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Simple AI Chat Endpoint
    """
    output = model(req.message, max_new_tokens=100)
    return {"response": output}
