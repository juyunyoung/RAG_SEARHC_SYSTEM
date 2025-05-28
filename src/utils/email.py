import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def send_email(to_email: str, subject: str, body: str) -> bool:
    try:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["To"] = to_email
        msg["From"] = os.getenv("SMTP_USERNAME")
        
        msg.attach(MIMEText(body, "plain"))
        
        with smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT"))) as server:
            server.starttls()
            server.login(os.getenv("SMTP_USERNAME"), os.getenv("SMTP_PASSWORD"))
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

def send_upload_notification(user_id: str, file_name: str):
    """Send email notification when document is uploaded"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg["From"] = os.getenv("SMTP_USERNAME")
        msg["To"] = user_id  # Assuming user_id is email
        msg["Subject"] = f"Document Upload Complete: {file_name}"
        
        # Create email body
        body = f"""
        Hello,
        
        Your document "{file_name}" has been successfully uploaded and processed.
        You can now ask questions about this document in the RAG Search System.
        
        Best regards,
        RAG Search System Team
        """
        
        msg.attach(MIMEText(body, "plain"))
        
        # Connect to SMTP server and send email
        with smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT"))) as server:
            server.starttls()
            server.login(os.getenv("SMTP_USERNAME"), os.getenv("SMTP_PASSWORD"))
            server.send_message(msg)
            
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        # Don't raise the exception to prevent blocking the upload process 