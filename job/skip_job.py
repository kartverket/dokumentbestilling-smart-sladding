import requests
import pdf_utils
from url_utils import database_base_url, api_base_url, model_base_url
import uuid
import logging


def hentDokumenterTilSladding():
    dokumenter = requests.get(f'{database_base_url()}/ubehandlede_dokumenter')

    if dokumenter.status_code != 200:
        logging.error(f'Kunne ikke hente ubehandlede dokumenter. Statuskode: {dokumenter.status_code}')
        raise Exception('Kunne ikke hente ubehandlede dokumenter')

    logging.info(f'Det er {len(dokumenter.json())} ubehandlede dokumenter')

    for dokument in dokumenter.json():
        dokumentaar = dokument.get('dokumentaar')
        dokumentnummer = dokument.get('dokumentnummer')
        embetenummer = dokument.get('embetenummer')

        dokumentStatus = requests.get(f'{database_base_url()}/dokumentstatus/{dokumentaar}/{dokumentnummer}/{embetenummer}')

        if dokumentStatus.status_code != 200:
            logging.error(f'Kunne ikke hente status for dokument: {dokumentaar}_{dokumentnummer}_{embetenummer}. Statuskode: {dokumentStatus.status_code}')
            continue

        if dokumentStatus.text != '"KLAR_FOR_BEHANDLING"':
            logging.info(f'Status fra dokument: {dokumentaar}_{dokumentnummer}_{embetenummer} er endret til {dokumentStatus.text}. Hopper over.')
            continue

        docid = f"{dokumentaar}_{dokumentnummer}_{embetenummer}"

        document_url = f'{api_base_url()}intern/pantebok/gjenpart/{docid}?attestering=false'

        logging.info(f'=== Starting processing for document: {docid} ===')
        logging.info(f'Document URL: {document_url}')
        logging.info(f'API Base URL configured as: {api_base_url()}')

        try:
            pdf_bytes = pdf_utils.get_pdf_bytes(document_url)
            logging.info(f'✓ Successfully retrieved PDF: {len(pdf_bytes)} bytes')
        except Exception as e:
            logging.error(f'✗ Failed to retrieve PDF for dokument: {docid}')
            logging.error(f'Error: {str(e)}')
            continue

        if not pdf_bytes or len(pdf_bytes) < 100:
            logging.error(f'✗ PDF for dokument {docid} is invalid: {len(pdf_bytes) if pdf_bytes else 0} bytes')
            if pdf_bytes:
                logging.error(f'PDF content (first 100 bytes): {pdf_bytes[:100]}')
            continue

        logging.info(f'✓ PDF validated: {len(pdf_bytes)} bytes')

        model_url = f'{model_base_url()}/model'
        logging.info(f'Model URL: {model_url}')
        logging.info(f'Model Base URL configured as: {model_base_url()}')
        logging.info(f'Sending {len(pdf_bytes)} bytes to model API...')

        try:
            response = requests.post(
                model_url,
                data=pdf_bytes,
                headers={
                    'Content-Type': 'application/pdf',
                    'Content-Length': str(len(pdf_bytes))
                },
                timeout=600  # 10 minute timeout for large PDFs
            )
            logging.info(f'✓ Model API response: status={response.status_code}')
            logging.info(f'Response headers: {dict(response.headers)}')
            logging.info(f'Response size: {len(response.content)} bytes')

            if response.status_code != 200:
                logging.error(f'✗ Model returned error status {response.status_code}')
                logging.error(f'Response body: {response.text[:500]}')
                continue

            sladdinger = response.json()
            logging.info(f'✓ Model processing complete: {len(sladdinger)} redactions found')
        except requests.exceptions.Timeout:
            logging.error(f'✗ Timeout calling model API (>600s)')
            continue
        except Exception as e:
            logging.error(f'✗ Error calling model API: {str(e)}')
            import traceback
            logging.error(f'Traceback: {traceback.format_exc()}')
            continue

        transformed_sladdinger = [
            {
                'id': str(uuid.uuid4()),
                'dokumentaar': dokumentaar,
                'dokumentnummer': dokumentnummer,
                'embetenummer': embetenummer,
                'sidetall': sladding.get('page'),
                'type': 'PERSONNUMMER',
                'height': sladding.get('height'),
                'width': sladding.get('width'),
                'x': sladding.get('x'),
                'y': sladding.get('y'),
                'mlGenerated': True,
                'mlStatus': ''
            }
            for i, sladding in enumerate(sladdinger)
        ]

        response = requests.put(f'{database_base_url()}/labels/{dokumentaar}/{dokumentnummer}/{embetenummer}?mlFerdigBehandlet=true', json=transformed_sladdinger)

        if response.status_code == 200:
            logging.info(f'Sendt sladdinger for dokument: {docid} til databasen og markert som mlFerdigBehandlet')
        else:
            logging.error(f'Kunne ikke sende sladdinger for dokument: {docid} til databasen. Statuskode: {response.status_code}')

if __name__ == '__main__':
    # Logging basic config, save log as json
    logging.basicConfig(
        level=logging.INFO,
        format= "{'time':'%(asctime)s', 'name': '%(name)s', 'level': '%(levelname)s', 'message': '%(message)s'}",
        handlers=[
            logging.StreamHandler()
        ]
    )

    hentDokumenterTilSladding()
