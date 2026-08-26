"""PDF fetching, kept separate from model_main.py so the job avoids the ML imports."""
import requests
import logging


def download_pdf(document_url: str) -> bytes:
    try:
        response = requests.get(document_url, timeout=300)

        if response.status_code != 200:
            if response.status_code == 401:
                try:
                    error_json = response.json()
                    if error_json.get('error') == 'Dokumentet er skjermet':
                        logging.info(f"Document is protected (skjermet), skipping.")
                        raise ValueError("SKJERMET_DOCUMENT")
                except (requests.exceptions.JSONDecodeError, KeyError):
                    pass

            logging.error(f"Failed to download PDF. HTTP Status Code: {response.status_code}")
            logging.error(f"Response headers: {dict(response.headers)}")
            raise ValueError(f"HTTP {response.status_code}: {response.text[:200]}")

        content_type = response.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower() and not response.content.startswith(b'%PDF'):
            logging.error(f"Downloaded content is not a PDF! Content-Type: {content_type}")
            raise ValueError(f"Response is not a PDF. Content-Type: {content_type}, Size: {len(response.content)} bytes")

        logging.info(f"PDF downloaded successfully: {len(response.content)} bytes")
        return response.content

    except requests.exceptions.Timeout:
        logging.error(f"Timeout downloading PDF from {document_url}")
        raise ValueError(f"Timeout downloading PDF from {document_url}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error downloading PDF: {str(e)}")
        raise ValueError(f"Request error: {str(e)}")


def get_pdf_bytes(document_url: str) -> bytes:
    """Reads the PDF from a local path, or downloads it if the URL is http(s)."""
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
