import streamlit as st
from auth.session import is_authenticated
from database.bigquery_client import BigQueryClient
from rag.document_processor import DocumentProcessor
from langchain.chains import RetrievalQA
from langchain_huggingface  import HuggingFaceEndpoint
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def render_main_content():
    """Render the main content area with document selection and Q&A"""
    if not is_authenticated():
        st.warning("Please log in to use the system")
        return
    
    # Initialize clients
    doc_processor = DocumentProcessor()
    db_client = BigQueryClient(
        dataset_id=os.getenv("BIGQUERY_DATASET")
    )
    # Get user's documents
    documents = db_client.get_user_documents(st.session_state.user_id)
    
    if documents.empty:
        st.info("Please upload a document to start")
        return
    
    # Document selection
    selected_doc = st.selectbox(
        "Select a document",
        documents["file_name"].tolist(),
        key="document_selector"
    )
    
    if selected_doc:
        # Get selected document's chunks
        doc_id = documents[documents["file_name"] == selected_doc]["document_id"].iloc[0]
        
        vector_store = db_client.get_database()
 
        
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_length=128,
            temperature=0.5,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        # Create QA chain with custom prompt
        template = """Answer the following question. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Question: {question}
        
        Answer:"""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["question"],
            template=template
        )
        # Create QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(metadata_filter={"document_id": doc_id}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        # Question input
        question = st.text_input("Ask a question about the document:")
        print("question:::"+question)
        if question:
            try:
                # Get answer
                result = qa_chain({"question": question})
                print("Result:", result)  # 디버깅을 위한 전체 결과 출력
                answer = result["answer"]
                
                # Display answer
                st.write("Answer:", answer)
                
                # Display sources
                with st.expander("View Sources"):
                    for doc in result["source_documents"]:
                        st.write(doc.page_content)
                        st.write("---")
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                print(f"Detailed error: {str(e)}")  # 디버깅을 위한 상세 에러 출력
    else:
        st.info("Please select a document to start.")

