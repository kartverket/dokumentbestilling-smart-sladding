# Use an official, lightweight Python runtime as a parent image
FROM python:3.14-slim

# Set environment variables to optimize Python performance inside the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to utilize Docker build caching
COPY requirements.txt .

# Install the specified Python packages
RUN pip install --no-cache-dir -r requirements.txt

#Install poppler
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

# Install Tesseract OCR
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev && rm -rf /var/lib/apt/lists/*

# Copy application code to the container
COPY app .

# Copy root files to the container
COPY . .

# Configure application to listen on port 8080
EXPOSE 8080

# Start Gunicorn with 4 worker processes
CMD ["sh", "-c", "gunicorn --config config/gunicorn_config_${MODE}.py -w 4 -b 0.0.0.0:8080 app:app"]
