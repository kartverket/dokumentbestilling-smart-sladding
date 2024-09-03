import requests
import model_main

api_base_url = 'http://localhost:8080/intern/pantebok/gjenpart'
database_base_url = 'http://localhost:8000/'

def hentDokumenterTilSladding():
    dokumenter = requests.get(f'{database_base_url}/ubehandlede_dokumenter')

    print(dokumenter.json())
    for dokument in dokumenter.json():
        dokumentaar = dokument.get('dokumentaar')
        dokumentnummer = dokument.get('dokumentnummer')
        embetenummer = dokument.get('embetenummer')

        docid = f"{dokumentaar}_{dokumentnummer}_{embetenummer}"

        sladdinger = model_main.main(docid, api_base_url)

        transformed_sladdinger = [
            {
                'dokumentaar': dokumentaar,
                'dokumentnummer': dokumentnummer,
                'embetenummer': embetenummer,
                'sidetall': sladding.get('page'),
                'index': i,
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
            requests.put(f'{database_base_url}/labels/{dokumentaar}/{dokumentnummer}/{embetenummer}', json=transformed_sladdinger)

        requests.patch(f'{database_base_url}/dokument_behandlet', json=dokument)


if __name__ == '__main__':
    hentDokumenterTilSladding()
