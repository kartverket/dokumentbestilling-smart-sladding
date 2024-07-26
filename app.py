from flask import Flask, jsonify, request
import model_main
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
base_url = os.getenv('DOKUMENT_URL', default='http://localhost:3000/pantebok')
#base_url = "https://dokumentbestilling-smart-sladding-manual.atkv3-dev.kartverket-intern.cloud/pantebok"

@app.route('/health')
def health():

    return jsonify(health="healthy")


@app.route('/model', methods=['GET'])
def get_bounding_boxes():

    docid = request.args.get("dokumentIdent", type=str)

    json_responses = model_main.main(docid, base_url)

    return jsonify(json_responses)


if __name__ == '__main__':
    app.run(host='localhost', port=5070)