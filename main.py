import streamlit as st
import fitz  # PyMuPDF
import chromadb
import google.generativeai as genai
import os
import datetime

# LangChain-related imports
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Fix PyTorch dynamic linking
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"
os.environ["STREAMLIT_SERVER_WATCH_DELAY"] = "3"

# Load sentence-transformers for embedding
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    HAS_SENTENCE_TRANSFORMERS = True
except Exception as e:
    st.error(f"Error loading SentenceTransformer: {str(e)}")
    HAS_SENTENCE_TRANSFORMERS = False
    embedding_model = None

# Setup ChromaDB
try:
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="pregnancy_docs")
except Exception as e:
    st.error(f"Error initializing ChromaDB: {str(e)}")
    collection = None

# LangChain Embedding Wrapper
lc_embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(client=chroma_client, collection_name="pregnancy_docs", embedding_function=lc_embedding_model)

# PDF Text Extraction
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
        embedding = embedding_model.encode(text).tolist()
        existing_ids = set(collection.get()["ids"])
        if doc_id in existing_ids:
            collection.delete(ids=[doc_id])
        collection.add(ids=[doc_id], documents=[text], embeddings=[embedding])
        return True
    except Exception as e:
        st.error(f"ChromaDB Insertion Error: {str(e)}")
        return False

# Custom QA Prompt Template with System Prompt
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

# LangChain + Gemini QA
def generate_response(query, conversation_history):
    try:
        if not st.session_state.get("api_key_configured", False):
            return "Please configure your Gemini API key in the sidebar first."
            
        # Get API key from session state
        api_key = st.session_state.get("gemini_api_key", "")
        if not api_key:
            return "API key not found in session state. Please reconfigure."

        # Create custom prompt
        custom_prompt = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        
        # Use custom prompt in QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": custom_prompt}
        )
        
        response = qa_chain.run(query)
        return response.strip()
    except Exception as e:
        st.error(f"LangChain QA Error: {str(e)}")
        return f"Error: {str(e)}"

# Generate unique doc ID
def generate_doc_id(filename):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{filename}_{timestamp}"

# Main UI
def main():
    st.set_page_config(page_title="RAG CHATBOT", page_icon="👶", layout="wide")
    st.title("Streamlit based RAG Application")
    st.markdown("Upload PDF and ask questions.")

    # Init session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "documents_uploaded" not in st.session_state:
        st.session_state.documents_uploaded = []
    if "api_key_configured" not in st.session_state:
        st.session_state.api_key_configured = False
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""

    # Sidebar: API + Upload
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
            if not collection:
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
            st.subheader("Processed File")
            for doc in st.session_state.documents_uploaded:
                st.write(f"- {doc}")
            if st.button("Clear All Document"):
                try:
                    ids = collection.get().get("ids", [])
                    if ids:
                        collection.delete(ids=ids)
                    st.session_state.documents_uploaded = []
                    st.success("Cleared all documents.")
                except Exception as e:
                    st.error(f"Clear Error: {str(e)}")

    # Chat and QA interface
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Ask Questions")
        with st.container(height=400, border=True):
            for i, message in enumerate(st.session_state.conversation_history):
                if i % 2 == 0:
                    st.chat_message("user").write(message)
                else:
                    st.chat_message("assistant").markdown(message)

    with col2:
        st.header("Instructions")
        st.markdown("""
        1. Get a Gemini API key from [Google AI Studio](https://ai.google.dev/)
        2. Paste and save your key in the sidebar
        3. Upload PDF of any domain you want
        4. Click "Process Documents"
        5. Ask anything questions about the uploaded documents
        """)

    query = st.chat_input("Ask some questions")
    if query:
        st.chat_message("user").write(query)
        st.session_state.conversation_history.append(query)

        if not st.session_state.api_key_configured:
            response = "Please configure your Gemini API key in the sidebar."
        elif not st.session_state.documents_uploaded:
            response = "Please upload a document first."
        elif not collection:
            response = "Database not initialized."
        else:
            with st.spinner("Retrieving and generating answer..."):
                response = generate_response(query, st.session_state.conversation_history)

        st.chat_message("assistant").markdown(response)
        st.session_state.conversation_history.append(response)

if __name__ == "__main__":
    main()