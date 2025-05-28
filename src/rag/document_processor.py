from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Any
import json
import datetime
import pandas as pd
import os
from docx import Document as DocxDocument
from PyPDF2 import PdfReader
import torch
import numpy as np

class DocumentProcessor:
    def __init__(self):
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize sentence transformer model
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model with proper error handling"""
        try:
            # Force CPU device
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.model.eval()
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def process_document(self, file_path: str, user_id: str) -> Dict[str, Any]:
        """Process document and return document data and chunks"""
        # Extract text based on file type
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            full_text = self._extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            full_text = self._extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            full_text = self._extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Split into chunks
        chunks = self.text_splitter.split_text(full_text)
        
        # Generate embeddings for chunks
        try:
            with torch.no_grad():
                chunk_embeddings = self.model.encode(chunks, convert_to_tensor=False)
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise
        
        # Create document metadata
        document_id = str(uuid.uuid4())
        document_data = {
            "document_id": document_id,
            "user_id": user_id,
            "file_name": os.path.basename(file_path),
            "upload_time": str(datetime.datetime.utcnow()),
            "meta_data": json.dumps({
                "num_chunks": len(chunks),
                "total_chars": len(full_text),
                "file_type": file_extension[1:]  # Remove the dot
            })
        }
        
        # Create chunks data
        chunks_data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_data = {
                "chunk_id": str(uuid.uuid4()),
                "document_id": document_id,
                "chunk_text": chunk,
                "embedding_vector": embedding.tolist(),
                "chunk_index": i
            }
            chunks_data.append(chunk_data)
        
        return {
            "document": document_data,
            "chunks": chunks_data
        }

    def get_relevant_chunks(self, question: str, chunks_df: pd.DataFrame, top_k: int = 3) -> List[str]:
        """Get most relevant chunks for a question"""
        try:
            # Get question embedding
            with torch.no_grad():
                question_embedding = self.model.encode(question, convert_to_tensor=False)
            
            # Calculate cosine similarity
            similarities = []
            for _, row in chunks_df.iterrows():
                chunk_embedding = row["embedding_vector"]
                similarity = self._cosine_similarity(question_embedding, chunk_embedding)
                similarities.append((row["chunk_text"], similarity))
            
            # Sort by similarity and get top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in similarities[:top_k]]
        except Exception as e:
            print(f"Error getting relevant chunks: {str(e)}")
            raise

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) 