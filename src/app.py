import streamlit as st
from ui.sidebar import render_sidebar
from ui.main_content import render_main_content
from ui.login import render_login_page
from auth.session import init_session, is_authenticated
from utils.config import load_config
from dotenv import load_dotenv
def main():
    load_dotenv()
    # Initialize session state
    init_session()
    
    # Set page config
    st.set_page_config(
        page_title="RAG Search System",
        page_icon="🔍",
        layout="wide"
    )
    
    # Check if user is authenticated
    if not is_authenticated():
        render_login_page()
    else:
        # Render sidebar
        render_sidebar()
        
        # Render main content
        render_main_content()

if __name__ == "__main__":
    main() 