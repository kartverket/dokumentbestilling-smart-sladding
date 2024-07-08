from flask import Flask, jsonify, request
import model_main

app = Flask(__name__)


@app.route('/model/', methods=['POST'])
def get_bounding_boxes():

    # Hent ute dokumentinfo(rmasjon fra request body, år, id, embete
    data = request.get_json()
    
    aar, id, embete = data.get('aar'), data.get('id'), data.get('embete')
    
    json_boxes = model_main.main(aar, id, embete)

    return jsonify(json_boxes)

if __name__ == '__main__':
    app.run(host='localhost', port=5070)