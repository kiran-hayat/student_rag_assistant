import streamlit as st
import os
from file_processor import FileProcessor
from rag_chain import RAGSystem
from utils import clear_directory, format_docs
import time
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
groq_api = os.getenv('GROQ_API_KEY')
os.environ['GROQ_API_KEY'] = groq_api  # Set the environment variable

# Page configuration
st.set_page_config(
    page_title="Study Assistant RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "rag_system" not in st.session_state:
    # TEMPORARY: Hardcoded API key (remove before sharing/deploying)
    
    
    st.session_state.rag_system = RAGSystem()
    st.session_state.processed = False
    st.session_state.uploaded_files = []
    st.session_state.docs = []

# ... [rest of your existing code remains the same] ...

# In the sidebar, remove the API key input completely:
with st.sidebar:
    st.title("📚 Study Assistant RAG")
    st.markdown("""
    Upload your study materials (PDFs, Word, PowerPoint) and get:
    - Summaries
    - Quizzes
    - Exam preparation
    - Explanations
    """)
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload your study materials",
        type=["pdf", "docx", "pptx", "ppt", "txt"],
        accept_multiple_files=True
    )
    
    # Process button
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Clear previous data
                clear_directory("data")
                if os.path.exists("vector_db"):
                    clear_directory("vector_db")
                
                # Process files
                processor = FileProcessor()
                st.session_state.docs = processor.process_files(uploaded_files)
                st.session_state.rag_system.initialize_vector_db(st.session_state.docs)
                st.session_state.rag_system.initialize_qa_chain()
                st.session_state.processed = True
                st.session_state.uploaded_files = [file.name for file in uploaded_files]
            
            st.success("Documents processed successfully!")
        else:
            st.warning("Please upload files first.")
    
    # Display processed files
    if st.session_state.processed:
        st.subheader("Processed Files")
        for file in st.session_state.uploaded_files:
            st.markdown(f"- {file}")
    
    # API key input
    
    st.session_state.rag_system.llm.groq_api_key = groq_api
    
    # Model settings
    st.subheader("Model Settings")
    model_name = st.selectbox(
    "Model",
    ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],  # Keep deprecated model last for legacy
    index=0  # Defaults to llama3-70b-8192
)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    if st.button("Update Model Settings"):
        st.session_state.rag_system.llm.model_name = model_name
        st.session_state.rag_system.llm.temperature = temperature
        st.success("Model settings updated!")

# Main content area
st.title("Study Assistant RAG")
st.markdown("Get help with your study materials using AI-powered retrieval and generation.")

# Task selection
task_type = st.radio(
    "Select Task Type",
    ["summary", "quiz", "exam", "explain"],
    format_func=lambda x: {
        "summary": "📝 Summary",
        "quiz": "❓ Generate Quiz",
        "exam": "📝 Create Exam",
        "explain": "💡 Explain Concept"
    }[x],
    horizontal=True
)

# Query input
query = st.text_area(
    "Enter your question or topic",
    placeholder="E.g., 'Summarize the key concepts in chapter 3' or 'Create a quiz about cellular biology'",
    height=100
)

# Submit button
if st.button("Get Assistance"):
    if not st.session_state.processed:
        st.warning("Please upload and process documents first.")
    elif not query.strip():
        st.warning("Please enter a question or topic.")
    else:
        with st.spinner("Generating response..."):
            start_time = time.time()
            response = st.session_state.rag_system.query(query, task_type)
            elapsed_time = time.time() - start_time
            
            st.subheader("Response")
            st.markdown(response)
            
            # Display performance metrics
            st.caption(f"Generated in {elapsed_time:.2f} seconds")

# Display document chunks if processed
if st.session_state.processed and st.checkbox("Show document chunks"):
    st.subheader("Document Chunks")
    st.markdown(format_docs(st.session_state.docs[:5]))  # Show first 5 chunks
    if len(st.session_state.docs) > 5:
        st.info(f"Showing 5 of {len(st.session_state.docs)} chunks. Upload more specific questions to see relevant chunks.")