import os
import hashlib
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_file_hash(file_path: str) -> str:
    """Generate a unique hash for a file."""
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

def split_documents(docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(docs)

def clear_directory(directory: str):
    """Clear all files in a directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def format_docs(docs: List[Document]) -> str:
    """Format documents for display."""
    return "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n\n{doc.page_content}" for doc in docs])