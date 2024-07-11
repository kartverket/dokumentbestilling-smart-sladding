from flask import Flask, jsonify, request, Response
import model_main

app = Flask(__name__)


@app.route('/health')
def health():

    return jsonify(health="healthy")


@app.route('/model', methods=['GET'])
def get_bounding_boxes():

    docid = request.args.get("dokumentIdent", type=str)

    json_responses = model_main.main(docid)

    return jsonify(json_responses)


if __name__ == '__main__':
    app.run(host='localhost', port=5070)