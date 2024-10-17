import requests
import model_main
from url_utils import database_base_url, api_base_url
import uuid

def hentDokumenterTilSladding():
    dokumenter = requests.get(f'{database_base_url()}/ubehandlede_dokumenter')

    if dokumenter.status_code != 200:
        print(f'Kunne ikke hente ubehandlede dokumenter. Statuskode: {dokumenter.status_code}')
        raise Exception('Kunne ikke hente ubehandlede dokumenter')

    print(f'Det er {len(dokumenter.json())} ubehandlede dokumenter')

    for dokument in dokumenter.json():
        dokumentaar = dokument.get('dokumentaar')
        dokumentnummer = dokument.get('dokumentnummer')
        embetenummer = dokument.get('embetenummer')

        docid = f"{dokumentaar}_{dokumentnummer}_{embetenummer}"

        url = f'{api_base_url()}intern/pantebok/gjenpart/{docid}'
        print(f'Kjører modell på: {url}')
        sladdinger = model_main.main(url)

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

        if (transformed_sladdinger != []):
            response = requests.put(f'{database_base_url()}/labels/{dokumentaar}/{dokumentnummer}/{embetenummer}', json=transformed_sladdinger)

            if response.status_code == 200:
                print(f'Sendt sladdinger for dokument: {docid} til databasen')
            else:
                print(f'Kunne ikke sende sladdinger for dokument: {docid} til databasen. Statuskode: {response.status_code}')

        response = requests.patch(f'{database_base_url()}/dokument_behandlet', json=dokument)

        if response.status_code == 200:
            print(f'Merket dokument: {docid} som behandlet')
        else:
            print(f'Kunne ikke merke dokument: {docid} som behandlet. Statuskode: {response.status_code}')


if __name__ == '__main__':
    hentDokumenterTilSladding()
