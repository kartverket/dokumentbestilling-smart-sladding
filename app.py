import requests
import model_main

base_url = 'http://localhost:8080/intern/pantebok/gjenpart'
database_base_url = 'http://localhost:8000/'

def hentDokumenterTilSladding():
    dokumenter = requests.get(f'{database_base_url}/ubehandlede_dokumenter')

    for dokument in dokumenter.json():
        dokumentaar = dokument.get('dokumentaar')
        dokumentnummer = dokument.get('dokumentnummer')
        embetenummer = dokument.get('embetenummer')

        docid = f"{dokumentaar}_{dokumentnummer}_{embetenummer}"

        sladdinger = model_main.main(docid, base_url)

        transformed_sladdinger = [
            {
                'dokumentaar': dokumentaar,
                'dokumentnummer': dokumentnummer,
                'embetenummer': embetenummer,
                'sidetall': sladding.get('page'),
                'index': 0,
                'type': 'PERSONNUMMER',
                'height': sladding.get('height'),
                'width': sladding.get('width'),
                'x': sladding.get('x'),
                'y': sladding.get('y'),
                'mlGenerated': True,
            }
            for sladding in sladdinger
        ]

        for sladding in sladdinger:
            requests.put(f'{database_base_url}/labels/{dokumentaar}/{dokumentnummer}/{embetenummer}', json=transformed_sladdinger)


if __name__ == '__main__':
    hentDokumenterTilSladding()
