from flask import Flask, jsonify
import requests

app = Flask(__name__)


@app.route('/model/', methods=['POST'])
def get_bounding_boxes():

    # Hent ute dokumentinfo(rmasjon fra request body, år, id, embete
    response = requests.get(url)
    
    #get from api

    # Laste ned dokuemntet fra BestillingsAPI via url som vi lager

    # Kjøre modellen på dokumentet og få tilbake bounding boxes


    return "Bbbs" # Returnere bounding boxes i json format

if __name__ == '__main__':
    app.run(host='localhost', port=5070)