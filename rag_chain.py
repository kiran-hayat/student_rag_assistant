from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()
groq_api = os.getenv('GROQ_API_KEY')
os.environ['GROQ_API_KEY'] = groq_api 
class RAGSystem:
    def __init__(self, model_name="llama3-70b-8192", temperature=0.7):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.llm = ChatGroq(
            temperature=temperature,
            model_name=model_name,
            groq_api_key= groq_api
        )
        self.vector_db = None
        self.retriever = None
        self.qa_chain = None
    
    def initialize_vector_db(self, docs, db_path="vector_db"):
        """Initialize or load FAISS vector database."""
        os.makedirs(db_path, exist_ok=True)
        
        if os.path.exists(os.path.join(db_path, "index.faiss")):
            self.vector_db = FAISS.load_local(db_path, self.embedding_model)
        else:
            self.vector_db = FAISS.from_documents(docs, self.embedding_model)
            self.vector_db.save_local(db_path)
        
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})
    
    def initialize_qa_chain(self):
        """Initialize the QA chain with custom prompts."""
        # Custom prompt templates
        summary_template = """You are a helpful study assistant. Use the following pieces of context to create a concise and accurate summary.
        The summary should cover all key points and concepts from the context.
        
        Context: {context}
        
        Summary:"""
        
        quiz_template = """You are a helpful study assistant. Create a quiz with 5-10 questions based on the following context.
        Include multiple choice questions, true/false questions, and short answer questions.
        Provide the correct answers at the end.
        
        Context: {context}
        
        Quiz:"""
        
        exam_template = """You are a helpful study assistant. Create a comprehensive exam based on the following context.
        The exam should include:
        - 5 multiple choice questions
        - 3 true/false questions
        - 2 short answer questions
        - 1 essay question
        Provide the correct answers and grading criteria.
        
        Context: {context}
        
        Exam:"""
        
        explain_template = """You are a helpful study assistant. Explain the following concept in simple terms and in detail.
        Break down complex ideas into smaller parts and use examples where appropriate.
        
        Context: {context}
        
        Explanation:"""
        
        answer_template = """You are a helpful study assistant. Answer the following question based on the context provided.
        Be thorough and detailed in your response.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        self.prompt_templates = {
            "summary": PromptTemplate.from_template(summary_template),
            "quiz": PromptTemplate.from_template(quiz_template),
            "exam": PromptTemplate.from_template(exam_template),
            "explain": PromptTemplate.from_template(explain_template),
            "answer": PromptTemplate.from_template(answer_template),
        }
    
    def query(self, question: str, task_type: str = "summary") -> str:
        """Query the RAG system based on task type."""
        if not self.retriever:
            return "Please upload and process documents first."
        
        # Get relevant documents
        docs = self.retriever.get_relevant_documents(question)
        
        if not docs:
            return "No relevant information found in the documents."
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Get the appropriate prompt template
        prompt = self.prompt_templates.get(task_type, self.prompt_templates["summary"])
        
        # Create chain
        chain = prompt | self.llm
        
        # Invoke chain
        response = chain.invoke({"context": context, "question": question})
        
        return response.content
    
    def update_model_settings(self, model_name: str, temperature: float):
        """Update the LLM model settings."""
        self.llm = ChatGroq(
            temperature=temperature,
            model_name=model_name,
            groq_api_key=groq_api
        )