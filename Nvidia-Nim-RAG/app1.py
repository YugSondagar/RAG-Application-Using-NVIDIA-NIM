import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv

load_dotenv()

nvidia_api_key = os.getenv("NVIDIA_API_KEY")
nvidia_embeddings_api_key = os.getenv("NVIDIA_EMBEDDINGS_API_KEY")

llm = ChatNVIDIA(model="meta/llama3-70b-instruct",api_key=nvidia_api_key)

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings(model='nvidia/llama-3.2-nemoretriever-300m-embed-v2',api_key=nvidia_embeddings_api_key)
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)

st.title("Nvidia NIM Chatbot")

prompt = ChatPromptTemplate.from_template(
"""
Anwer the question based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Question: {input}

"""
)

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Embeddings"):
    vector_embeddings()
    st.write("Vector Store DB is ready!")

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    start = time.process_time()
    reponse = retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(reponse['answer'])

    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(reponse['context']):
            st.write(doc.page_content)
            st.write("--------------------------")
