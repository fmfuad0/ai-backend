from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from ctransformers import AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import os

app = FastAPI()

# Load lightweight model
MODEL_PATH = "tinyllama.gguf"
if not os.path.exists(MODEL_PATH):
    import requests
    url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    with requests.get(url, stream=True) as r:
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

model = AutoModelForCausalLM.from_pretrained(".", model_file=MODEL_PATH, model_type="llama")

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = None
file_embeddings = []
file_contents = []

# ------------------ Models ------------------
class EditRequest(BaseModel):
    code: str
    instruction: str

class ChatRequest(BaseModel):
    message: str

# ------------------ Endpoints ------------------
@app.get("/")
def root():
    return {"status": "running"}

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    global index, file_embeddings, file_contents
    file_contents = []
    file_embeddings = []

    for f in files:
        content = (await f.read()).decode("utf-8")
        file_contents.append({"name": f.filename, "content": content})
        emb = embed_model.encode(content)
        file_embeddings.append(emb)

    # Build FAISS index
    dim = len(file_embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(file_embeddings).astype("float32"))

    return {"status": "files indexed", "count": len(files)}

@app.post("/search")
def search_code(query: str):
    global index, file_contents
    if index is None:
        return {"error": "No files uploaded"}

    q_emb = embed_model.encode(query).astype("float32")
    D, I = index.search(np.array([q_emb]), k=3)
    results = [{"file": file_contents[i]["name"], "content": file_contents[i]["content"]} for i in I[0]]
    return {"results": results}

@app.post("/edit")
def edit(req: EditRequest):
    prompt = f"""
You are an expert coding assistant.

Instruction:
{req.instruction}

Code:
{req.code}
"""
    out = model(prompt, max_new_tokens=200)
    return {"edited_code": out}

@app.post("/chat")
def chat(req: ChatRequest):
    out = model(req.message, max_new_tokens=150)
    return {"response": out}
