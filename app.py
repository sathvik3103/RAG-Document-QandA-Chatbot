import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions asked only and only based on the context provided.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("/Users/sathvik/Desktop/Langchain/RAG Document Q&A/pdfs")
        
        # Check if directory exists and contains PDFs
        if not os.path.isdir("/pdfs"):
            st.error("The specified PDF directory does not exist.")
            return
        
        st.session_state.docs = st.session_state.loader.load()
        
        if not st.session_state.docs:
            st.error("No PDF documents found in the specified directory.")
            return
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        if not st.session_state.final_documents:
            st.error("No text could be extracted from the PDF documents.")
            return
        
        try:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Vector Database is ready to serve!")
        except Exception as e:
            st.error(f"An error occurred while creating the vector database: {str(e)}")

user_prompt = st.text_input("Enter your query about the Business Models and Infrastructure Requirements of the Indian Quick Commerce:")

if st.button("Document Embedding"):
    create_vector_embedding()

if user_prompt:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        print(f"Response Time: {time.process_time() - start} ")

        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('-------------------------')
    else:
        st.error("Vector database is not initialized. Please ensure documents are embedded first.")