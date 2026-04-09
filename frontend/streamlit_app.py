import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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

def process_document(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    if not text.strip():
        raise ValueError("No text could be extracted from this PDF")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore, len(chunks)

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.session_state.get("processed_file") != uploaded_file.name:
        with st.spinner("Processing document..."):
            try:
                vectorstore, count = process_document(uploaded_file)
                st.session_state.vectorstore = vectorstore
                st.session_state.chunk_count = count
                st.session_state.processed_file = uploaded_file.name
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    st.success(
        f"Successfully ingested {st.session_state.get('chunk_count', 0)} chunks!"
    )

question = st.text_input("Ask a question about this document")

if st.button("Get Answer") and question:
    if "vectorstore" not in st.session_state:
        st.error("Please upload a PDF first!")
    else:
        with st.spinner("Thinking..."):
            try:
                llm = ChatOpenAI(model="gpt-4", temperature=0)
                prompt = PromptTemplate(
                    template=PROMPT_TEMPLATE,
                    input_variables=["context", "question"]
                )
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    ),
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True
                )
                result = chain({"query": question})
                st.subheader("Answer")
                st.write(result["result"])
                with st.expander("Source Passages"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.markdown(
                            f"**Source {i+1}:** {doc.page_content[:200]}"
                        )
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")