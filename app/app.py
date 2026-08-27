import logging
from flask import Flask, jsonify, request
import model_main
import vlm_verifier
import zipped_timed_rotating_file_handler
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
base_url = os.getenv('DOKUMENT_URL', default='http://localhost:3000/pantebok')

# Where the application log goes and how many days of history to keep. Set by
# compose; the defaults are the container paths.
ML_LOG_DIR = os.getenv('ML_LOG_DIR', '/data/ml_logs')
LOG_BACKUP_DAYS = int(os.getenv('LOG_BACKUP_DAYS', '30'))

# The VLM verifier, off unless SLADD_VLM=1 and an endpoint is configured. Built
# once, not per request: it holds no state beyond the settings.
VLM = vlm_verifier.config_from_env()


@app.route('/health')
def health():
    return jsonify(health="healthy")


@app.route('/model', methods=['POST'])
def get_bounding_boxes():

    if not request.data:
        return jsonify({'error': 'No data provided in the request body'}), 400

    try:
        elektronisk_tinglyst = request.args.get('elektronisk_tinglyst', 'false').lower() == 'true'
        # Comma-separated XX_YYY codes from the grunnbok, e.g.
        # ?rettsstiftelsestyper=SR_JOU,SR_BSK enables per-document-type rule
        # profiles (see KOORDFAM_CODES in config). Omitted/empty = global.
        rettsstiftelsestyper = [k.strip() for k in
                                request.args.get('rettsstiftelsestyper', '')
                                .split(',') if k.strip()]
        # ?vlm=false turns the verifier off for one request even when the
        # container runs with it on, so the two can be compared on the same
        # document without a redeploy. It can never turn it on: without an
        # endpoint there is nothing to call.
        vlm = VLM if request.args.get('vlm', 'true').lower() != 'false' else None
        pdf_file_stream = request.get_data()
        bounding_boxes_result = model_main.run_model_on_pdf_bytes(
            pdf_file_stream, elektronisk_tinglyst=elektronisk_tinglyst,
            rettsstiftelsestyper=rettsstiftelsestyper, vlm=vlm)

        return jsonify(bounding_boxes_result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        zipped_timed_rotating_file_handler.ZippedTimedRotatingFileHandler(
            os.path.join(ML_LOG_DIR, "app.log"), when="midnight", backupCount=LOG_BACKUP_DAYS),
        logging.StreamHandler()
    ]
)

if __name__ == '__main__':
    # For development only - use gunicorn for production
    app.run(host='localhost', port=5070, debug=True)
