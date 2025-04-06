import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredFileLoader
)
from utils import get_file_hash, split_documents

class FileProcessor:
    def __init__(self, upload_dir="data"):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to disk and return its path."""
        file_path = os.path.join(self.upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a document based on its file type."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif file_ext in [".pptx", ".ppt"]:
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            # Try unstructured loader for other file types
            loader = UnstructuredFileLoader(file_path)
        
        return loader.load()
    
    def process_files(self, uploaded_files) -> List[Document]:
        """Process multiple uploaded files."""
        all_docs = []
        
        for uploaded_file in uploaded_files:
            try:
                # Save file
                file_path = self.save_uploaded_file(uploaded_file)
                
                # Load document
                docs = self.load_document(file_path)
                
                # Add source metadata
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["file_hash"] = get_file_hash(file_path)
                
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error processing file {uploaded_file.name}: {e}")
                continue
        
        # Split documents into chunks
        if all_docs:
            all_docs = split_documents(all_docs)
        
        return all_docs