"""
SafeLand AI — Email Alert Module
Handles sending email alerts via Gmail SMTP for landing analysis events,
zone selections, re-analysis notifications, and test emails.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Attempt to load .env file if python-dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_email_credentials() -> tuple[str, str, str]:
    """
    Retrieve email credentials from environment variables.
    Returns (sender, password, receiver).
    """
    sender = os.environ.get("SAFELAND_EMAIL_SENDER", "").strip()
    password = os.environ.get("SAFELAND_EMAIL_PASSWORD", "").strip()
    receiver = os.environ.get("SAFELAND_EMAIL_RECEIVER", "").strip()
    return sender, password, receiver


def is_email_configured() -> bool:
    """
    Check if all required email credentials are set in environment.
    """
    sender, password, receiver = get_email_credentials()
    return bool(sender and password and receiver)


def send_email_alert(subject: str, body: str, recipient: str = None) -> dict:
    """
    Sends an email alert using Gmail SMTP.

    Args:
        subject (str): The subject line of the email.
        body (str): The body text of the email.
        recipient (str, optional): The target email address. Defaults to SAFELAND_EMAIL_RECEIVER.

    Returns:
        dict: {
            "success": bool,
            "message": str
        }
    """
    sender, password, default_receiver = get_email_credentials()
    target_receiver = (recipient if recipient else default_receiver).strip()

    if not sender or not password or not target_receiver:
        missing = []
        if not sender:
            missing.append("SAFELAND_EMAIL_SENDER")
        if not password:
            missing.append("SAFELAND_EMAIL_PASSWORD")
        if not target_receiver:
            missing.append("SAFELAND_EMAIL_RECEIVER")
        return {
            "success": False,
            "message": f"Email credentials not configured (Missing: {', '.join(missing)})"
        }

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = target_receiver
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain", "utf-8"))

        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()

        return {
            "success": True,
            "message": "Email sent successfully"
        }

    except Exception as e:
        raw_err = str(e)
        # Safe non-secret error formatting (sanitize password if present in trace)
        if password and password in raw_err:
            safe_err = raw_err.replace(password, "********")
        else:
            safe_err = raw_err
        return {
            "success": False,
            "message": f"SMTP Error: {safe_err}"
        }
