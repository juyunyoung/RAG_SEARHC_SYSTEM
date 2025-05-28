import streamlit as st
from database.bigquery_client import BigQueryClient
from auth.session import login_user
from utils.password import verify_password, hash_password
import os

def render_login_page():
    st.title("Login")
    
    with st.form("login_form"):
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        password='a'
        user_id ='a'
        if submit_button:
            # Initialize BigQuery client            
            client = BigQueryClient(
                dataset_id=os.getenv("BIGQUERY_DATASET")
            )
            
            # Get user from database
            user = client.get_user(user_id)
            hash_passowrd= hash_password(password)
          
            if user and verify_password(password, user["password"]):
                # Update last login timestamp
                client.update_last_login(user_id)
                
                # Login user
                login_user(user_id)
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid user_id or password")
    
    # Add registration link
    st.markdown("Don't have an account? [Register here](/register)") 