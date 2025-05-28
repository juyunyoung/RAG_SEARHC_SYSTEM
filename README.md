# RAG Search System

A Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions about them.

## Features

- User authentication and session management
- Document upload and parsing using UpstageDocumentParseLoader
- Document chunking and embedding generation
- Vector-based semantic search
- Question answering using LLM
- Email notifications for document processing
- Document management (view/delete)

## Tech Stack

- Framework: LangChain
- UI: Streamlit
- Database: BigQuery
- Document Parser: UpstageDocumentParseLoader
- Embeddings: HuggingFace Embeddings
- LLM: HuggingFace Hub (flan-t5-large)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.streamlit/secrets.toml`:
```toml
# Authentication
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# BigQuery
GOOGLE_CLOUD_PROJECT = "your-project-id"
BIGQUERY_DATASET = "rag_system"

# Email
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "your-email@gmail.com"
SMTP_PASSWORD = "your-app-password"

# Upstage API
UPSTAGE_API_KEY = "your-upstage-api-key"
```

3. Set up Google Cloud credentials:
- Create a service account
- Download the JSON key file
- Set the environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

4. Run the application:
```bash
streamlit run src/app.py
```

## Usage

1. Log in to the system
2. Upload a document (PDF, DOCX, or TXT)
3. Wait for processing and email notification
4. Select a document from the sidebar
5. Ask questions about the document
6. View answers with source references

## Project Structure

```
src/
├── app.py                 # Main application
├── auth/
│   └── session.py        # Authentication and session management
├── database/
│   └── bigquery_client.py # BigQuery client
├── rag/
│   └── document_processor.py # Document processing and RAG pipeline
├── ui/
│   ├── sidebar.py        # Sidebar UI component
│   └── main_content.py   # Main content UI component
└── utils/
    ├── config.py         # Configuration management
    └── email.py          # Email utilities
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
## 가상실행
.\venv\Scripts\activate
python 3.10.6