import streamlit as st
from auth.session import is_authenticated, logout_user
from database.bigquery_client import BigQueryClient
from rag.document_processor import DocumentProcessor
from utils.email import send_upload_notification
import os

def render_sidebar():
    """Render the sidebar with file upload and document list"""
    with st.sidebar:
        st.title("RAG Search System")
        
        if not is_authenticated():
            st.warning("Please log in to use the system")
            return
        
        # File upload section
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt"],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp/{uploaded_file.name}"
            os.makedirs("temp", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process document
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    try:
                        
                        # Initialize clients
                        db_client = BigQueryClient(dataset_id=os.getenv("BIGQUERY_DATASET"))
                        doc_processor = DocumentProcessor()
                        
                        # Process document
                        result = doc_processor.process_document(
                            temp_path,
                            st.session_state.user_id
                        )
                        
                        # Save to database using single transaction
                        success = db_client.save_document_with_chunks(
                            result["document"],
                            result["chunks"]
                        )
                        
                        if success:
                            # Send notification
                            # send_upload_notification(
                            #     st.session_state.user_id,
                            #     uploaded_file.name
                            # )
                            st.success("Document processed successfully!")
                        else:
                            st.error("Failed to save document and chunks")
                        
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
        
        # Document list section
        st.subheader("Your Documents")
        try:
            db_client = BigQueryClient(dataset_id=os.getenv("BIGQUERY_DATASET"))
            documents = db_client.get_user_documents(st.session_state.user_id)
            
            if documents.empty:
                st.info("No documents uploaded yet")
            else:
                for _, doc in documents.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📄 {doc['file_name']}")
                    with col2:
                        if st.button("🗑️", key=f"delete_{doc['document_id']}"):
                            try:
                                db_client.delete_document(doc['document_id'])
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting document: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
        
        # Logout button
        if st.button("Logout"):
            logout_user()
            st.rerun() 