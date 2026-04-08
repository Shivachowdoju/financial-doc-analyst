import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

st.title("AI Financial Document Analyst")
st.markdown("Upload a financial PDF and ask questions about it")

PROMPT_TEMPLATE = """
You are a financial analyst assistant.
Use only the context below to answer the question.
If the answer is not in the context, say 'I cannot find this in the document.'
Always cite which part of the document your answer comes from.

Context: {context}
Question: {question}
Answer:"""

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def process_document(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(chunks, embeddings)
    return vectorstore, len(chunks)

def get_answer(vectorstore, question):
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    result = chain({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:200] for doc in result["source_documents"]]
    }

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing document..."):
        text = extract_text(uploaded_file)
        vectorstore, chunk_count = process_document(text)
        st.session_state.vectorstore = vectorstore
    st.success(f"Successfully ingested {chunk_count} chunks!")

question = st.text_input("Ask a question about this document")

if st.button("Get Answer") and question:
    if "vectorstore" not in st.session_state:
        st.error("Please upload a PDF first!")
    else:
        with st.spinner("Thinking..."):
            result = get_answer(st.session_state.vectorstore, question)
        st.subheader("Answer")
        st.write(result["answer"])
        with st.expander("Source Passages"):
            for i, source in enumerate(result["sources"]):
                st.markdown(f"**Source {i+1}:** {source}")


