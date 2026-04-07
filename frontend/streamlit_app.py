import streamlit as st
import requests
API_URL = "http://localhost:8000"

st.title("AI Financial Document Analyst")
st.markdown("Upload a financial PDF and ask questions about it")

# File upload

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file:
    with st.spinner("Ingesting document..."):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": uploaded_file.getvalue()}
        )
    if response.status_code == 200:
        st.success(response.json()["message"])
    else:
        st.error(f"Upload failed! Status: {response.status_code}")
        st.error(f"Details: {response.text}")

# Question input
question = st.text_input("Ask a question about this document")

# Answer section
if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question}
        )
    data = response.json()
    st.subheader("Answer")
    st.write(data["answer"])
    with st.expander("Source Passages"):
        for i, source in enumerate(data["sources"]):
            st.markdown(f"**Source {i+1}:** {source}")


