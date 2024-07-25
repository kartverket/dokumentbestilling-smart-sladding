# Use the base Python 3.11.9-slim image
FROM python:3.11.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies including Tesseract, Poppler, and Norwegian language package
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        libleptonica-dev \
        poppler-utils \
        build-essential \
        pkg-config \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /smart_sladding_ml

# Copy your application code into the container
COPY . /smart_sladding_ml

# Install Python dependencies
COPY requirements.txt /smart_sladding_ml/
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

#Download Language files
#RUN python -c "import easyocr; easyocr.Reader(['no', 'en', 'da'], model_storage_directory='/smart_sladding_ml/easyocr_models')"

EXPOSE 5070

# Specify the command to run your application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5070"]