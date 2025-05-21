# Import only what we need
import sys
import os
import datetime
import streamlit as st
import fitz  

st.set_page_config(page_title="RAG CHATBOT", page_icon="👶", layout="wide")
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    print("pysqlite3 not found; using system sqlite3")

import google.generativeai as genai

# LangChain imports
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# PyTorch dynamic linking fix
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"
os.environ["STREAMLIT_SERVER_WATCH_DELAY"] = "3"

try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    HAS_SENTENCE_TRANSFORMERS = True
except Exception as e:
    st.error(f"Error loading SentenceTransformer: {str(e)}")
    HAS_SENTENCE_TRANSFORMERS = False
    embedding_model = None

lc_embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
from langchain.vectorstores import Chroma

try:
    vectordb = Chroma(
        persist_directory="./chroma_db",
        collection_name="pregnancy_docs",
        embedding_function=lc_embedding_model
    )
    collection = vectordb._collection  
except Exception as e:
    st.error(f"Error initializing ChromaDB: {str(e)}")
    vectordb = None
    collection = None


def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        return "".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"PDF Extraction Error: {str(e)}")
        return ""

# Add to ChromaDB
def add_text_to_chroma(text, doc_id):
    try:
        if vectordb is None:
            st.error("Vector database not initialized.")
            return False
            
        # Using LangChain's add_texts method instead of direct ChromaDB access
        vectordb.add_texts(
            texts=[text],
            ids=[doc_id],
            metadatas=[{"source": doc_id}]
        )
        return True
    except Exception as e:
        st.error(f"ChromaDB Insertion Error: {str(e)}")
        return False

# Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
SYSTEM: You are a helpful assistant that provides information from the provided documents. 
Follow these rules strictly:
1. You are a helpful assistant specializing in given information by the user.
2. Use bullet points, markdown, and highlight key advice clearly.
3. Give citations properly with the document name and page number.
4. If the query is out of context then explain that you are not able to answer as it is out of the scope.
5. Try to elaborate the answer as much as possible.

CONTEXT: {context}

QUESTION: {question}

YOUR RESPONSE (in bullet points, strictly based on the provided context):
"""

# Gemini + LangChain QA
def generate_response(query, conversation_history):
    try:
        if not st.session_state.get("api_key_configured", False):
            return "Please configure your Gemini API key in the sidebar first."

        api_key = st.session_state.get("gemini_api_key", "")
        if not api_key:
            return "API key not found in session state. Please reconfigure."
            
        # Check if LangChain GoogleGenerativeAI is available
        # if not 'LANGCHAIN_GENAI_AVAILABLE' in globals() or not LANGCHAIN_GENAI_AVAILABLE:
        #     return "Error: langchain_google_genai module is not available. Please ensure it's installed properly."

        # Setup Prompt
        custom_prompt = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # Import here to ensure it's available
        # from langchain_google_genai import GoogleGenerativeAI
        llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": custom_prompt}
        )

        return qa_chain.run(query).strip()
    except Exception as e:
        st.error(f"LangChain QA Error: {str(e)}")
        return f"Error: {str(e)}"

def generate_doc_id(filename):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{filename}_{timestamp}"

# Main App
def main():
    st.title("Welcome to the Pregnancy Guidance Chatbot")
    st.markdown("Upload PDFs and ask questions.")

    # Session state init
    for key in ["conversation_history", "documents_uploaded", "api_key_configured", "gemini_api_key"]:
        if key not in st.session_state:
            st.session_state[key] = [] if "history" in key or "documents" in key else False if "configured" in key else ""

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your Gemini API key", type="password")
        if api_key and st.button("Save API Key"):
            try:
                genai.configure(api_key=api_key)
                st.session_state.api_key_configured = True
                st.session_state.gemini_api_key = api_key
                st.success("API Key configured!")
            except Exception as e:
                st.error(f"API Key Error: {str(e)}")    
                st.session_state.api_key_configured = False

        if st.session_state.api_key_configured:
            st.success("API Key Status: Configured ✓")
        else:
            st.error("API Key Status: Not Configured ✗")
            st.info("Get a key from https://ai.google.dev/")
        st.divider()

        st.header("Upload PDF")
        uploaded_files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            if vectordb is None:
                st.error("ChromaDB not ready.")
            elif st.button("Process Document"):
                with st.spinner("Processing..."):
                    for pdf_file in uploaded_files:
                        if pdf_file.name in st.session_state.documents_uploaded:
                            st.info(f"Already processed: {pdf_file.name}")
                            continue
                        text = extract_text_from_pdf(pdf_file)
                        if text:
                            doc_id = generate_doc_id(pdf_file.name)
                            if add_text_to_chroma(text, doc_id):
                                st.session_state.documents_uploaded.append(pdf_file.name)
                                st.success(f"Uploaded: {pdf_file.name}")

        if st.session_state.documents_uploaded:
            st.subheader("Processed Files")
            for doc in st.session_state.documents_uploaded:
                st.write(f"- {doc}")
            if st.button("Clear All Document"):
                try:
                    # Using LangChain's method to delete all
                    vectordb.delete(vectordb.get().get("ids", []))
                    st.session_state.documents_uploaded = []
                    st.success("Cleared all documents.")
                except Exception as e:
                    st.error(f"Clear Error: {str(e)}")

    # Main chat UI
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Ask Questions")
        with st.container(height=400, border=True):
            for i, message in enumerate(st.session_state.conversation_history):
                (st.chat_message("user") if i % 2 == 0 else st.chat_message("assistant")).write(message)

    with col2:
        st.header("Instructions")
        st.markdown("""
        1. Get a Gemini API key from [Google AI Studio](https://ai.google.dev/)
        2. Paste and save your key in the sidebar
        3. Upload one or more PDFs
        4. Click "Process Document"
        5. Ask questions based on uploaded content
        """)

    query = st.chat_input("Ask some questions")
    if query:
        st.chat_message("user").write(query)
        st.session_state.conversation_history.append(query)

        if not st.session_state.api_key_configured:
            response = "Please configure your Gemini API key in the sidebar."
        elif not st.session_state.documents_uploaded:
            response = "Please upload a document first."
        elif vectordb is None:
            response = "Database not initialized."
        else:
            with st.spinner("Retrieving and generating answer..."):
                response = generate_response(query, st.session_state.conversation_history)

        st.chat_message("assistant").markdown(response)
        st.session_state.conversation_history.append(response)

if __name__ == "__main__":
    main()
