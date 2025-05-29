from google.cloud import bigquery
from typing import List, Dict, Any
from langchain_google_community import BigQueryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from google.oauth2 import service_account
import pandas as pd
import os
import json
import io
import numpy as np

class BigQueryClient:
    
    def __init__(self, dataset_id: str):        

        self.credentials = service_account.Credentials.from_service_account_file(os.getenv("SERVICE_ACCOUNT_FILE"))     
        self.client = bigquery.Client(credentials=self.credentials, project=self.credentials.project_id)                
        self.dataset_id = dataset_id
        self.users_table = f"{self.credentials.project_id}.{dataset_id}.USER"
        self.documents_table = f"{self.credentials.project_id}.{dataset_id}.DOCUMENT"
        self.chunks_table = f"{self.credentials.project_id}.{dataset_id}.DOCUMENT_CHUNK"
        self.qa_table = f"{self.credentials.project_id}.{dataset_id}.QA_HISTORY"
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.database = BigQueryVectorStore(
                project_id=self.credentials.project_id,
                dataset_name=self.dataset_id,
                table_name="DOCUMENT_CHUNK",
                location="asia-northeast3",
                embedding=self.embeddings,
                credentials=self.credentials,
                embedding_dimension=384  # all-MiniLM-L6-v2 모델의 출력 차원
        )



    def get_user(self, user_id: str) -> dict:
        """Get user by user_id"""
        query = f"""
        SELECT *
        FROM `{self.users_table}`
        WHERE user_id = @user_id
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        
        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()
        
        for row in results:
            return {
                "user_id": row.user_id,
                "password": row.password,             
                "last_login": row.last_login
            }
        return None
    
    def create_user(self, user_id: str, password: str) -> bool:
        """Create a new user"""
        query = f"""
        INSERT INTO `{self.users_table}` (user_id, password, last_login)
        VALUES (@user_id, @password, CURRENT_TIMESTAMP())
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("password", "STRING", password)
            ]
        )
        
        try:
            self.client.query(query, job_config=job_config)
            return True
        except Exception as e:
            print(f"Error creating user: {str(e)}")
            return False
    
    def update_last_login(self, user_id: str) -> bool:
        """Update user's last login timestamp"""
        query = f"""
        UPDATE `{self.users_table}`
        SET last_login = CURRENT_TIMESTAMP()
        WHERE user_id = @user_id
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        
        try:
            self.client.query(query, job_config=job_config)
            return True
        except Exception as e:
            print(f"Error updating last login: {str(e)}")
            return False

    def save_document(self, document_data: Dict[str, Any]):
        """Save document metadata"""
        try:
            # Start a transaction
            with self.client.transaction() as transaction:
                # Insert document
                rows_to_insert = [document_data]
                errors = self.client.insert_rows_json(
                    self.documents_table, 
                    rows_to_insert,
                    transaction=transaction
                )
                if errors:
                    raise Exception(f"Error inserting document: {errors}")
                
                # If we get here, the transaction will be committed
                return True
        except Exception as e:
            print(f"Error in save_document: {str(e)}")
            return False

    def save_chunks(self, chunks_data: List[Dict[str, Any]]):
        """Save document chunks with embeddings"""
        try:
            # Start a transaction
            with self.client.transaction() as transaction:
                # Insert chunks
                # errors = self.client.insert_rows_json(
                #     self.chunks_table, 
                #     chunks_data,
                #     transaction=transaction
                # )

                # all_texts = [chunk['chunk_text'] for chunk in chunks_data]
                # metadatas = [{"source": str(i)} for i in range(len(all_texts))]
                # self.database.add_texts(all_texts, metadatas=metadatas);
                self.database.save_document_with_chunks(chunks_data)


                if errors:
                    raise Exception(f"Error inserting chunks: {errors}")
                
                # If we get here, the transaction will be committed
                return True
        except Exception as e:
            print(f"Error in save_chunks: {str(e)}")
            return False

    def save_document_with_chunks(self, document_data: Dict[str, Any], chunks_data: List[Dict[str, Any]]) -> bool:
        """Save document and its chunks in a single transaction"""
        try:
            # Convert embedding vectors to array format
            for chunk in chunks_data:
                if isinstance(chunk['embedding'], (list, np.ndarray)):
                    # Convert to list if it's numpy array
                    if isinstance(chunk['embedding'], np.ndarray):
                        chunk['embedding'] = chunk['embedding'].tolist()
                    # Ensure it's a list of floats
                    chunk['embedding'] = [float(x) for x in chunk['embedding']]
            
            # First try to save chunks
            chunk_errors = self.client.insert_rows_json(
                self.chunks_table, 
                chunks_data
            )
            
            if chunk_errors:
                print(f"Error inserting chunks: {chunk_errors}")
                raise Exception(f"Error inserting chunks: {chunk_errors}")

            # Then try to save document
            doc_rows = [document_data]
            doc_errors = self.client.insert_rows_json(
                self.documents_table, 
                doc_rows
            )
            if doc_errors:
                print(f"Error inserting document: {doc_errors}")
                raise Exception(f"Error inserting document: {doc_errors}")
            
            return True
        except Exception as e:
            print(f"Error in save_document_with_chunks: {str(e)}")
            return False

    def get_user_documents(self, user_id: str) -> pd.DataFrame:
        """Get all documents for a user"""
        query = f"""
        SELECT *
        FROM `{self.documents_table}`
        WHERE user_id = @user_id
        ORDER BY upload_time DESC
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        return self.client.query(query, job_config=job_config).to_dataframe()

    def get_document_chunks(self, document_id: str) -> pd.DataFrame:
        """Get all chunks for a document"""
        query = f"""
        SELECT *
        FROM `{self.chunks_table}`
        WHERE document_id = @document_id
        ORDER BY chunk_index
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("document_id", "STRING", document_id)
            ]
        )
        df = self.client.query(query, job_config=job_config).to_dataframe()
        
        # Convert embedding vectors to numpy arrays if needed
        if not df.empty and 'embedding' in df.columns:
            df['embedding'] = df['embedding'].apply(lambda x: np.array(x) if isinstance(x, list) else x)
        
        return df

    def delete_document(self, document_id: str):
        """Delete document and its chunks"""
        # Delete chunks first
        chunks_query = f"""
        DELETE FROM `{self.documents_table}`
        WHERE document_id = @document_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("document_id", "STRING", document_id)
            ]
        )
        self.client.query(chunks_query, job_config=job_config).result()

        # Delete document
        doc_query = f"""
        DELETE FROM `{self.documents_table}`
        WHERE document_id = @document_id
        """
        self.client.query(doc_query, job_config=job_config).result() 

    def get_database(self) -> BigQueryVectorStore:
        # Initialize embeddings
        return self.database

    def save_qa_history(self, user_id: str, document_id: str, question: str, answer: str) -> bool:
        """Save Q&A history to database"""
        try:
            # 현재 시간을 문자열로 변환
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 데이터 준비
            qa_data = {
                "user_id": user_id,
                "document_id": document_id,
                "question": question,
                "answer": answer,
                "timestemp": current_time
            }
            
            print(f"Saving QA history: {qa_data}")  # 디버깅을 위한 로그
            
            # 데이터 저장
            errors = self.client.insert_rows_json(
                self.qa_table,
                [qa_data]
            )
            
            if errors:
                print(f"Error saving QA history: {errors}")
                return False
                
            print("Successfully saved QA history")  # 디버깅을 위한 로그
            return True
            
        except Exception as e:
            print(f"Error in save_qa_history: {str(e)}")
            return False

    def get_qa_history(self, user_id: str, document_id: str = None) -> pd.DataFrame:
        """Get Q&A history for a user and optionally for a specific document"""
        query = f"""
        SELECT *
        FROM `{self.qa_table}`
        WHERE user_id = @user_id
        """
        
        params = [bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
        
        if document_id:
            query += " AND document_id = @document_id"
            params.append(bigquery.ScalarQueryParameter("document_id", "STRING", document_id))
            
        query += " ORDER BY timestemp DESC"
        
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        return self.client.query(query, job_config=job_config).to_dataframe()
