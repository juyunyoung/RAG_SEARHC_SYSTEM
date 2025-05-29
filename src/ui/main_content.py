import streamlit as st
from auth.session import is_authenticated
from database.bigquery_client import BigQueryClient
from rag.document_processor import DocumentProcessor
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceEndpoint

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
 
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            convert_system_message_to_human=True
        )
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        # Create QA chain with custom prompt
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        # Create QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(metadata_filter={"document_id": doc_id}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": QA_CHAIN_PROMPT,
                "document_variable_name": "context"
            }
        )
        
        # Question input
        question = st.text_input("Ask a question about the document:")
        if len(question) <= 5 and question:
            st.warning("Please enter a longer question (more than 5 characters)")
            question = ""
        print("question:::"+question)
        if question:
            try:
                # Get answer
                result = qa_chain.invoke({"question": question})                
                answer = result["answer"]
                
                # Save Q&A history
                try:
                    save_success = db_client.save_qa_history(
                        user_id=st.session_state.user_id,
                        document_id=doc_id,
                        question=question,
                        answer=answer
                    )
                    
                    if not save_success:
                        st.warning("Failed to save Q&A history")
                except Exception as e:
                    print(f"Error saving Q&A history: {str(e)}")
                    st.warning("Failed to save Q&A history")
                
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

        # Display Q&A history
        st.subheader("Recent Q&A History")
        try:
            qa_history = db_client.get_qa_history(
                user_id=st.session_state.user_id,
                document_id=doc_id
            )
            
            if not qa_history.empty:
                for _, qa in qa_history.iterrows():
                    with st.expander(f"Q: {qa['question']} ({qa['timestemp']})"):
                        st.write("A:", qa['answer'])
            else:
                st.info("No Q&A history yet")
        except Exception as e:
            st.error(f"Error loading Q&A history: {str(e)}")
    else:
        st.info("Please select a document to start.")

