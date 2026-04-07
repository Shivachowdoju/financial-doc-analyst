import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

def extract_text_from_pdf(pdf_path:str) -> str:
    doc = fitz.open(pdf_path)
    text =""
    for page in doc:
        text += page.get_text()
    return text
    
def ingest_document(pdf_path: str):
    raw_text = extract_text_from_pdf(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()

    vector_store = Chroma.from_texts(
        chunks,
        embeddings,
        persist_directory="./vectorstore/chroma_db"
    )

    return f"Ingested {len(chunks)} chunks successfully."