from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.ingest import ingest_document
from app.query import get_answer
import shutil
import os

#create FastAPI app

app = FastAPI(title="Financial Document Analyst API")

#Data model for incoming question

class QuestionRequest(BaseModel):
    question: str

# upload pdf

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    save_path = f"./data/uploads/{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = ingest_document(save_path)
    return {"message": result}

# ask question

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    result = get_answer(request.question)
    return result

# Health check 

@app.get("/health")
async def health_check():
    return {"status": "ok"} 

