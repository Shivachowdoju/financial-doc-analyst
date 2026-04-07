from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.ingest import ingest_document
from app.query import get_answer
import shutil
import os

app = FastAPI(title="Financial Document Analyst API")

# Ensure upload directory exists
os.makedirs("./data/uploads", exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF document."""
    try:
        if not file.filename.lower().endswith(".pdf"):
           raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        save_path = f"./data/uploads/{file.filename}"
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = ingest_document(save_path)
        return {"message": result}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Answer a question about the uploaded document."""
    try:
        result = get_answer(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}