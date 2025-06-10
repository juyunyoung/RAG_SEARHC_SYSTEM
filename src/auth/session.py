import streamlit as st
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import Optional
from utils.config import Environment as env

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def init_session():
    """Initialize session state variables"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "token" not in st.session_state:
        st.session_state.token = None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, env.SECRET_KEY, algorithm=env.ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

def login_user(user_id: str):
    """Set user as logged in"""
    st.session_state.user_id = user_id
    st.session_state.authenticated = True
    token = create_access_token({"sub": user_id})
    st.session_state.token = token

def logout_user():
    """Log out user"""
    st.session_state.user_id = None
    st.session_state.authenticated = False
    st.session_state.token = None

def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return st.session_state.authenticated 