from typing import Dict, Any
from dotenv import load_dotenv
import os


class Environment:
    _instance = None
    _initialized = False

    load_dotenv()
    GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
    BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET") 
    SMTP_SERVER = os.getenv("SMTP_SERVER")
    SECRET_KEY = os.getenv("SECRET_KEY")
    ALGORITHM = os.getenv("ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Environment, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True

