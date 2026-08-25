import requests
import pdf_utils
from url_utils import database_base_url, api_base_url, model_base_url
import uuid
import logging


def hentDokumenterTilSladding():
    response = requests.get(f'{database_base_url()}/ubehandlede_dokumenter')

    if response.status_code != 200:
        logging.error(f'Could not fetch unprocessed documents. Status code: {response.status_code}')
        raise Exception('Could not fetch unprocessed documents')

    logging.info(f'There are {len(response.json())} unprocessed documents')

    for document in response.json():
        ident = document.get('dokumentIdent')
        dokumentaar = ident.get('dokumentaar')
        dokumentnummer = ident.get('dokumentnummer')
        embetenummer = ident.get('embetenummer')

        docid = f"{dokumentaar}_{dokumentnummer}_{embetenummer}"

        er_elektronisk_tinglyst = document.get('erElektroniskTinglyst')
        if er_elektronisk_tinglyst:
            logging.info(f'Document {docid} is elektronisk tinglyst')

        dokumentStatus = requests.get(f'{database_base_url()}/dokumentstatus/{dokumentaar}/{dokumentnummer}/{embetenummer}')

        if dokumentStatus.status_code != 200:
            logging.error(f'Could not fetch status for document: {dokumentaar}_{dokumentnummer}_{embetenummer}. Status code: {dokumentStatus.status_code}')
            continue

        if dokumentStatus.text != '"KLAR_FOR_BEHANDLING"':
            logging.info(f'Status for document: {dokumentaar}_{dokumentnummer}_{embetenummer} changed to {dokumentStatus.text}. Skipping.')
            continue

        document_url = f'{api_base_url()}intern/pantebok/gjenpart/{docid}?attestering=false'

        logging.info(f'Running model on document: {document_url}')

        try:
            pdf_bytes = pdf_utils.get_pdf_bytes(document_url)
        except ValueError as e:
            if "SKJERMET_DOCUMENT" in str(e):
                logging.info(f'Document {docid} is protected (skjermet), skipping')
                continue
            logging.error(f'Failed to retrieve PDF for dokument: {docid}')
            logging.error(f'Error: {str(e)}')
            continue
        except Exception as e:
            logging.error(f'Failed to retrieve PDF for dokument: {docid}')
            logging.error(f'Error: {str(e)}')
            continue

        model_url = f'{model_base_url()}/model'

        try:
            response = requests.post(
                model_url,
                params={'elektronisk_tinglyst': str(er_elektronisk_tinglyst).lower()},
                data=pdf_bytes,
                headers={
                    'Content-Type': 'application/pdf',
                    'Content-Length': str(len(pdf_bytes))
                },
                timeout=600  # large PDFs
            )

            if response.status_code != 200:
                logging.error(f'Model returned error status {response.status_code}')
                logging.error(f'Response body: {response.text[:500]}')
                continue

            sladdinger = response.json()
            logging.info(f'Model processing complete: {len(sladdinger)} redactions found')
        except requests.exceptions.Timeout:
            logging.error(f'Timeout calling model API (>600s)')
            continue
        except Exception as e:
            logging.error(f'Error calling model API: {str(e)}')
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
            logging.info(f'Sent sladdinger for document: {docid} to the database, marked mlFerdigBehandlet')
        else:
            logging.error(f'Could not send sladdinger for document: {docid} to the database. Status code: {response.status_code}')

if __name__ == '__main__':
    # JSON-shaped log lines, for the log collector.
    logging.basicConfig(
        level=logging.INFO,
        format= "{'time':'%(asctime)s', 'name': '%(name)s', 'level': '%(levelname)s', 'message': '%(message)s'}",
        handlers=[
            logging.StreamHandler()
        ]
    )

    hentDokumenterTilSladding()
