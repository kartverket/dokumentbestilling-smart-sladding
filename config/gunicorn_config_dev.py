import os

# Get the directory of this config file
basedir = os.path.abspath(os.path.dirname(__file__))
log_dir = os.path.join(basedir, '/data/gunicorn_logs/')

# Ensure logs directory exists
os.makedirs(log_dir, exist_ok=True)

# Worker configuration
worker_class = 'sync'
workers = 4
timeout = 600

# Logging configuration
loglevel = 'debug'
accesslog = os.path.join(log_dir, 'gunicorn_access_dev.log')
accesslogformat = "%(h)s %(l)s %(u)s %(t)s %(r)s %(s)s %(b)s %(f)s %(a)s"
errorlog = os.path.join(log_dir, 'gunicorn_error_dev.log')
capture_output = True
