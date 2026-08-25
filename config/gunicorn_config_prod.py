import os
import sys

# This file is read before gunicorn puts the app folder on sys.path, and
# logconfig_dict below refers to the handler class by dotted name.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Where the logs go. Set by compose; the default is the container path.
log_dir = os.getenv("GUNICORN_LOG_DIR", "/data/gunicorn_logs")
os.makedirs(log_dir, exist_ok=True)

# Days of history to keep per log file.
backup_days = int(os.getenv("LOG_BACKUP_DAYS", "30"))

# Worker configuration
worker_class = 'sync'
workers = 1
timeout = 600

# Logging configuration
loglevel = os.getenv("GUNICORN_LOGLEVEL", "debug")

# Must be set to something for gunicorn to log access at all. logconfig_dict
# below swaps out the handler, so none of this actually reaches stdout.
accesslog = "-"
errorlog = "-"

# Was named "accesslogformat", which is not a gunicorn setting, so the format
# was silently ignored and the default used. This is the correct name.
access_log_format = "%(h)s %(l)s %(u)s %(t)s %(r)s %(s)s %(b)s %(f)s %(a)s"

# capture_output dup2's fd 1 and 2 to the errorlog file at kernel level,
# bypassing the rotation set up below so the file would grow unbounded. Off:
# stray stdout goes to the docker log, which compose rotates.
capture_output = False

# Access and error logs through the same rotating zip handler as the
# application log. Without this gunicorn rotates nothing and
# gunicorn_access_prod.log grows until the disk is full.
logconfig_dict = {
    "version": 1,
    # dictConfig must not disable gunicorn's own loggers.
    "disable_existing_loggers": False,
    "formatters": {
        "rå": {"format": "%(message)s"},
        "tidsstemplet": {"format": "%(asctime)s [%(levelname)s] %(message)s"},
    },
    "handlers": {
        "access_fil": {
            "class": "zipped_timed_rotating_file_handler.ZippedTimedRotatingFileHandler",
            "filename": os.path.join(log_dir, "gunicorn_access_prod.log"),
            "when": "midnight",
            "backupCount": backup_days,
            "formatter": "rå",
        },
        "error_fil": {
            "class": "zipped_timed_rotating_file_handler.ZippedTimedRotatingFileHandler",
            "filename": os.path.join(log_dir, "gunicorn_error_prod.log"),
            "when": "midnight",
            "backupCount": backup_days,
            "formatter": "tidsstemplet",
        },
        "stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "tidsstemplet",
        },
    },
    "loggers": {
        "gunicorn.access": {
            "handlers": ["access_fil"],
            "level": "INFO",
            "propagate": False,
        },
        # The error log also goes to stdout so "./deploy.sh logs prod" still
        # shows startup and errors.
        "gunicorn.error": {
            "handlers": ["error_fil", "stdout"],
            "level": loglevel.upper(),
            "propagate": False,
        },
    },
}
