from flask import Flask, jsonify, request, Response
# import model_main

app = Flask(__name__)


@app.route('/model', methods=['GET'])
def get_bounding_boxes():

    # Hent ute dokumentinfo(rmasjon fra request body, år, id, embete
    data = request.args.get("dokumentIdent", type=str)
    print("data", data)
    
    aar, id, embete = data.split("-")

    print(aar, id, embete)
    
    # json_boxes = model_main.main(aar, id, embete)
    label_list = [{"x":100, "y":100, "height":100, "width":100, "page":1}, {"x":100, "y":100, "height":100, "width":100, "page":1}]
    # return jsonify(label_list)
    test = jsonify(label_list)
    # print("list:",test, type(test))

    # test.status_code=200
    return test

    # resp = Response(response=jsonify(label_list), status=200, mimetype="application/json")


    # return resp
    # val x: Double,
    # val y: Double,
    # val height: Double,
    # val width: Double,
    # val page: Int,

    # return jsonify(json_boxes)

if __name__ == '__main__':
    app.run(host='localhost', port=5070)