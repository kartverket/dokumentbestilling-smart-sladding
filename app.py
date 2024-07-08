from flask import Flask, jsonify, request
import model_main

app = Flask(__name__)


@app.route('/model/', methods=['POST'])
def get_bounding_boxes():

    # Hent ute dokumentinfo(rmasjon fra request body, år, id, embete
    data = request.get_json()
    
    docid = data.get('docid')
    
    json_boxes = model_main.main(docid)

    return jsonify(json_boxes)

if __name__ == '__main__':
    app.run(host='localhost', port=5070)