from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

generator = pipeline(
    "text-generation",
    model="Salesforce/codegen-350M-mono",
    device=-1
)

class Query(BaseModel):
    code: str
    instruction: str

@app.post("/edit")
def edit_code(query: Query):
    prompt = f"""
You are a coding assistant.

Modify the given code based on instruction.

Return ONLY the updated code.

Instruction:
{query.instruction}

Code:
{query.code}
"""

    result = generator(prompt, max_length=400, temperature=0.3)
    output = result[0]["generated_text"]

    # crude cleanup
    if "Code:" in output:
        output = output.split("Code:")[-1]

    return {"edited_code": output.strip()}