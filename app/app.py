import logging
from flask import Flask, jsonify, request
import model_main
import zipped_timed_rotating_file_handler
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
base_url = os.getenv('DOKUMENT_URL', default='http://localhost:3000/pantebok')


@app.route('/health')
def health():
    return jsonify(health="healthy")


@app.route('/model', methods=['POST'])
def get_bounding_boxes():

    if not request.data:
        return jsonify({'error': 'No data provided in the request body'}), 400

    try:
        pdf_file_stream = request.get_data()
        bounding_boxes_result = model_main.run_model_on_pdf_bytes(pdf_file_stream)

        return jsonify(bounding_boxes_result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        zzipped_timed_rotating_file_handler.ZippedTimedRotatingFileHandler("/data/ml_logs/app.log", when="midnight", backupCount=30),
        logging.StreamHandler()
    ]
)

if __name__ == '__main__':
    # For development only - use gunicorn for production
    app.run(host='localhost', port=5070, debug=True)
