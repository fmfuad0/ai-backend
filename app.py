from fastapi import FastAPI
from pydantic import BaseModel
from ctransformers import AutoModelForCausalLM

app = FastAPI()

# Load small GGUF model (very important)
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/TinyLlama-1.1B-Chat-GGUF",
    model_file="tinyllama-1.1b-chat.Q2_K.gguf",  # low memory version
    model_type="llama",
    gpu_layers=0
)

class EditRequest(BaseModel):
    code: str
    instruction: str

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/chat")
def chat(req: ChatRequest):
    output = model(req.message, max_new_tokens=150)
    return {"response": output}

@app.post("/edit")
def edit(req: EditRequest):
    prompt = f"""
You are a coding assistant.

Modify code based on instruction.

Instruction:
{req.instruction}

Code:
{req.code}
"""

    output = model(prompt, max_new_tokens=200)

    return {"edited_code": output}
