import json
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

        rettsstiftelsestyper = document.get('rettsstiftelsestyper')
        filrevisjonid = document.get('filRevisjonId')
        if not rettsstiftelsestyper:
            logging.warning(f'Document {docid} has no rettsstiftelsestyper, no rule profile will apply')

        model_url = f'{model_base_url()}/model'

        try:
            response = requests.post(
                model_url,
                params={
                    'elektronisk_tinglyst': str(er_elektronisk_tinglyst).lower(),
                    'rettsstiftelsestyper': rettsstiftelsestyper,
                    'filrevisjonid': filrevisjonid,
                },
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

            model_result = response.json()
            sladdinger = model_result['boxes']
            pipeline_version = model_result.get('pipeline_version')
            roterte_sider = [side for side in model_result.get('pages', [])
                             if side.get('rotation')]
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
                'mlStatus': '',
                'mlKilde': sladding.get('kilde'),
                'mlConf': sladding.get('yolo_conf'),
                'mlPipelineVersjon': pipeline_version
            }
            for i, sladding in enumerate(sladdinger)
        ]

        response = requests.put(
            f'{database_base_url()}/labels/{dokumentaar}/{dokumentnummer}/{embetenummer}',
            params={'mlFerdigBehandlet': 'true', 'mlPipelineVersjon': pipeline_version},
            json=transformed_sladdinger)

        if response.status_code == 200:
            logging.info(f'Sent sladdinger for document: {docid} to the database, marked mlFerdigBehandlet')
        else:
            logging.error(f'Could not send sladdinger for document: {docid} to the database. Status code: {response.status_code}')

        # Sent even when empty: the PUT replaces everything for the
        # filrevisjon, so a rerun also clears existing side-funn.
        side_funn = [
            {'sidetall': side['page'], 'funnType': 'ROTERT_SIDE', 'verdi': str(side['rotation'])}
            for side in roterte_sider
        ]
        funn_response = requests.put(f'{database_base_url()}/ml-side-funn/{filrevisjonid}', json=side_funn)

        if funn_response.status_code != 200:
            logging.error(f'Could not send side-funn for document: {docid}. Status code: {funn_response.status_code}')
        elif side_funn:
            logging.info(f'Flagged {len(side_funn)} rotated page(s) for document: {docid}')


class JsonFormatter(logging.Formatter):
    """Real JSON, one object per line, so the log collector can parse it."""

    def format(self, record):
        return json.dumps({
            'time': self.formatTime(record),
            'name': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
        }, ensure_ascii=False)


if __name__ == '__main__':
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    hentDokumenterTilSladding()
