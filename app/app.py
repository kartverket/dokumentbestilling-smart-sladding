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


def _log_safe(value, max_len=100):
    """A query-string value as one log line.

    max_len is in characters. A newline in a parameter would otherwise split
    the line in two and let the second half imitate a real log entry.
    """
    text = " ".join(str(value).split()) if value else ""
    return text[:max_len] if text else "none"


@app.route('/health')
def health():
    return jsonify(health="healthy")


def _vlm_log(stats):
    """The verifier's part of the request log line, empty when it was off."""
    v = stats.get("vlm")
    if not v:
        return ''
    if not v.get("judged"):
        return ', vlm=nothing to judge'
    return f', vlm={v["dropped"]}/{v["judged"]} removed'



_PHASE_ORDER = ("render", "orientation", "ocr", "yolo+match", "vlm",
                "postprocessing")


def _time_log(stats):
    """The phase seconds as one log segment, empty without timings."""
    t = stats.get("timings") or {}
    if not t:
        return ''
    phases = ([p for p in _PHASE_ORDER if p in t]
              + sorted(set(t) - set(_PHASE_ORDER)))
    return ', t=' + ' '.join(f'{p} {t[p]:.1f}s' for p in phases)


def _warn_on_vlm_failure(filrevisjonid, stats):
    """A verifier that answered nothing must not pass as one that said «ja».

    Every failure path keeps the box, so the document is still safe; the point
    is that a dead endpoint would otherwise leave no trace at all.
    """
    v = stats.get("vlm") or {}
    failed = sum(n for reason, n in (v.get("not_cached") or {}).items()
                 if reason in ("call failed", "breaker open"))
    if failed:
        logging.warning(
            f'Document {_log_safe(filrevisjonid)}: VLM answered nothing for '
            f'{failed} of {v.get("judged", 0)} boxes — they keep their sladd, '
            f'but the verifier is doing less than it looks like')


@app.route('/model', methods=['POST'])
def get_bounding_boxes():

    if not request.data:
        return jsonify({'error': 'No data provided in the request body'}), 400

    filrevisjonid = request.args.get('filrevisjonid') or None

    try:
        elektronisk_tinglyst = request.args.get('elektronisk_tinglyst', 'false').lower() == 'true'
        # Comma-separated XX_YYY codes from the grunnbok, e.g.
        # ?rettsstiftelsestyper=SR_JOU,SR_BSK enables per-document-type rule
        # profiles (see KOORDFAM_CODES in config). Omitted/empty = global.
        # getlist, because a caller that sends the codes as a JSON array gets
        # one param per code, and args.get would keep only the first.
        rettsstiftelsestyper = [
            code.strip()
            for value in request.args.getlist('rettsstiftelsestyper')
            for code in value.split(',') if code.strip()]
        # ?vlm=false turns the verifier off for one request even when the
        # container runs with it on, so the two can be compared on the same
        # document without a redeploy. It can never turn it on: without an
        # endpoint there is nothing to call.
        vlm = VLM if request.args.get('vlm', 'true').lower() != 'false' else None
        pdf_file_stream = request.get_data()
        stats = {}
        bounding_boxes_result = model_main.run_model_on_pdf_bytes(
            pdf_file_stream, name=filrevisjonid,
            elektronisk_tinglyst=elektronisk_tinglyst,
            rettsstiftelsestyper=rettsstiftelsestyper, vlm=vlm, stats=stats)

        logging.info(f'Document {_log_safe(filrevisjonid)}: '
                     f'{len(bounding_boxes_result)} boxes, rettsstiftelsestyper='
                     f'{_log_safe(",".join(rettsstiftelsestyper))}'
                     f'{_vlm_log(stats)}{_time_log(stats)}')
        _warn_on_vlm_failure(filrevisjonid, stats)
        return jsonify(bounding_boxes_result)

    except Exception as e:
        logging.exception(f'Document {_log_safe(filrevisjonid)}: model failed')
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
