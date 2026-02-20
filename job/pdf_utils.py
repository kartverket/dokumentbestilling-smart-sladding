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
    logging.info(f"Attempting to download PDF from: {document_url}")

    try:
        response = requests.get(document_url, timeout=300)  # 5 minute timeout

        logging.info(f"Download response: status={response.status_code}, "
                    f"content-type={response.headers.get('Content-Type')}, "
                    f"content-length={len(response.content)} bytes")

        if response.status_code != 200:
            # Check if document is protected/restricted (skjermet)
            if response.status_code == 401:
                try:
                    error_json = response.json()
                    if error_json.get('error') == 'Dokumentet er skjermet':
                        logging.info(f"Document is protected (skjermet), skipping: {document_url}")
                        raise ValueError("SKJERMET_DOCUMENT")  # Special marker for skipping
                except (requests.exceptions.JSONDecodeError, KeyError):
                    pass  # Not JSON or different error, continue with normal error handling

            logging.error(f"Failed to download PDF. HTTP Status Code: {response.status_code}")
            logging.error(f"Response headers: {dict(response.headers)}")
            logging.error(f"Response body (first 500 chars): {response.text[:500]}")
            raise ValueError(f"HTTP {response.status_code}: {response.text[:200]}")

        # Validate it's actually a PDF
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower() and not response.content.startswith(b'%PDF'):
            logging.error(f"Downloaded content is not a PDF! Content-Type: {content_type}")
            logging.error(f"Content (first 100 bytes): {response.content[:100]}")
            raise ValueError(f"Response is not a PDF. Content-Type: {content_type}, Size: {len(response.content)} bytes")

        if len(response.content) < 100:
            logging.error(f"PDF file too small: {len(response.content)} bytes")
            logging.error(f"Content: {response.content}")
            raise ValueError(f"PDF file too small: {len(response.content)} bytes")

        logging.info(f"PDF downloaded successfully: {len(response.content)} bytes")
        return response.content

    except requests.exceptions.Timeout:
        logging.error(f"Timeout downloading PDF from {document_url}")
        raise ValueError(f"Timeout downloading PDF from {document_url}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error downloading PDF: {str(e)}")
        raise ValueError(f"Request error: {str(e)}")


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
