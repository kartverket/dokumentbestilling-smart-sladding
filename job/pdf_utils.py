"""
Lightweight utilities for fetching PDF files.
Extracted from model_main.py to avoid importing heavy ML dependencies.
"""
import requests
import logging


def download_pdf(document_url: str) -> bytes:
    """
    Download a PDF file from a URL.

    Parameters:
    document_url (str): The URL of the PDF document.

    Returns:
    bytes: The content of the PDF file.
    """
    response = requests.get(document_url)

    if response.status_code == 200:
        logging.info("PDF downloaded successfully.")
    else:
        logging.error(f"Failed to download file. HTTP Status Code: {response.status_code}")

    return response.content


def get_pdf_bytes(document_url: str) -> bytes:
    """
    Retrieve PDF bytes from a local file path or a remote URL.

    Parameters:
        document_url (str): The URL or local path to the PDF document.

    Returns:
        bytes: The PDF file as bytes.

    Raises:
        ValueError: If the document URL is invalid or the download fails.
    """
    try:
        if document_url.lower().startswith('http'):
            pdf_bytes = download_pdf(document_url)
            if not pdf_bytes:
                raise ValueError(f"Failed to download PDF from URL: {document_url}")
        else:
            with open(document_url, 'rb') as file:
                pdf_bytes = file.read()
        return pdf_bytes
    except Exception as e:
        raise ValueError(f"Error retrieving PDF bytes: {e}")
