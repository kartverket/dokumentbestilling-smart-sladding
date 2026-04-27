#!/bin/bash
# Start the Flask app with Gunicorn (production server)
cd "$(dirname "$0")"

# Set environment variable to fix macOS fork issue
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Add the current directory to Python path for module imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

gunicorn --bind 0.0.0.0:5070 --config gunicorn_config.py app:app
