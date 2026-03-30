from fastapi import FastAPI
from pydantic import BaseModel
from ctransformers import AutoModelForCausalLM

app = FastAPI()

MODEL_PATH = "tinyllama.gguf"

# Load model — now assumes it exists
print("Loading TinyLlama model...")
model = AutoModelForCausalLM.from_pretrained(
    ".", model_file=MODEL_PATH, model_type="llama"
)
print("Model loaded!")

class EditRequest(BaseModel):
    code: str
    instruction: str

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/edit")
def edit(req: EditRequest):
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
    output = model(req.message, max_new_tokens=100)
    return {"response": output}
