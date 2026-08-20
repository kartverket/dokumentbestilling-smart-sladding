import os
import sys

# Config-filen leses før gunicorn legger app-mappen på sys.path, og
# logconfig_dict under refererer handler-klassen med punktnotasjon.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hvor loggene havner. Settes av compose; defaulten er containerstien.
log_dir = os.getenv("GUNICORN_LOG_DIR", "/data/gunicorn_logs")
os.makedirs(log_dir, exist_ok=True)

# Antall døgn historikk å beholde per loggfil.
backup_days = int(os.getenv("LOG_BACKUP_DAYS", "30"))

# Worker configuration
worker_class = 'sync'
workers = 1
timeout = 600

# Logging configuration
loglevel = os.getenv("GUNICORN_LOGLEVEL", "debug")

# Må være satt til noe for at gunicorn i det hele tatt skal logge
# tilgang. logconfig_dict under bytter ut handleren, så ingenting av
# dette havner faktisk på stdout.
accesslog = "-"
errorlog = "-"

# Het «accesslogformat» før. Det er ikke et gunicorn-navn, så formatet
# ble stille ignorert og defaulten brukt. Riktig navn er dette.
access_log_format = "%(h)s %(l)s %(u)s %(t)s %(r)s %(s)s %(b)s %(f)s %(a)s"

# capture_output dup2'er fd 1 og 2 til errorlog-filen på kernel-nivå.
# Det omgår rotasjonen vi setter opp under, og filen ville vokst uten
# tak. Av: stray stdout går til docker-loggen, som compose roterer.
capture_output = False

# Access- og error-loggen gjennom samme roterende zip-handler som
# applikasjonsloggen. Uten dette roterer gunicorn ingenting, og
# gunicorn_access_prod.log vokser til disken er full.
logconfig_dict = {
    "version": 1,
    # Gunicorn sine egne loggere må ikke slås av av dictConfig.
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
        # Error-loggen går også til stdout, så «./deploy.sh logs prod»
        # fortsatt viser oppstart og feil.
        "gunicorn.error": {
            "handlers": ["error_fil", "stdout"],
            "level": loglevel.upper(),
            "propagate": False,
        },
    },
}
