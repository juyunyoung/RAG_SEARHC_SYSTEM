from typing import Dict, Any
from dotenv import load_dotenv
import os


def load_config() -> Dict[str, Any]:
    load_dotenv()
    """Load configuration from Streamlit secrets"""
    return {
        "GOOGLE_CLOUD_PROJECT": os.getenv["GOOGLE_CLOUD_PROJECT"],
        "BIGQUERY_DATASET": os.getenv["BIGQUERY_DATASET"],
        "SMTP_SERVER": os.getenv["SMTP_SERVER"],
        "smtp_port": os.getenv["SMTP_PORT"],
        "smtp_username": os.getenv["SMTP_USERNAME"],
        "smtp_password": os.getenv["SMTP_PASSWORD"],
        "secret_key": os.getenv["SECRET_KEY"],
        "algorithm": os.getenv["ALGORITHM"],
        "access_token_expire_minutes": os.getenv["ACCESS_TOKEN_EXPIRE_MINUTES"],
        "upstage_api_key": os.getenv["UPSTAGE_API_KEY"],
        "openai_api_key": os.getenv["OPENAI_API_KEY"],
        "service_account_file": os.getenv["SERVICE_ACCOUNT_FILE"]
    } 