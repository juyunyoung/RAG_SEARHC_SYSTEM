import streamlit as st
from database.bigquery_client import BigQueryClient
from utils.password import verify_password
from auth.session import login_user

def render_login_page():
    """Render the login page"""
    st.title("Login")
    
    with st.form("login_form"):
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if not user_id or not password:
                st.error("Please enter both user_id and password")
                return
            
            try:
                client = BigQueryClient.get_instance()
                user = client.get_user(user_id)
                
                if user and verify_password(password, user["password"]):
                    login_user(user_id)
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid user_id or password")
            except Exception as e:
                st.error(f"Error during login: {str(e)}")
    
    # Add registration link
    st.markdown("Don't have an account? [Register here](/register)") 