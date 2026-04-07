from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv  

PROMPT_TEMPLATE = """

You are a financial analyst assistant.
Use only the context to answer the question.
If the answer is not in the context, say 'I cannot find in this document.'

context:
{context}

question:
{question}

Answer:
"""

def get_answer(question:str) -> str:

    #step 1 : Load the vector database from disk
    embeddings = OpenAIEmbeddings()
    vector_store=Chroma(persist_directory="./vectorstore/chroma_db", embedding_function=embeddings)

    #step 2 : set up the LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    #step 3 : Create the prompt
    prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

    #step 4 : Build the RAG Chain
    chain = RetrievalQA.from_chain_type(llm=llm, 
                                        retriever=vector_store.as_retriever( search_kwargs={"k":3}),
                                        chain_type_kwargs={"prompt": prompt}, return_source_documents=True)
    
    #step 5: Run the chain

    result = chain({"query": question})

    # Step 6: Return answer and sources
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:200] for doc in result["source_documents"]]
    }
    



