FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
  ffmpeg \
  libgl1 \
  libglib2.0-0 \
  curl \
  gcc \
  g++ \
  python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Install execstack explicitly since it's missing in newer Debian repos
RUN curl -L -o /usr/bin/execstack https://github.com/waseemzahid/execstack-static/raw/main/execstack && \
  chmod +x /usr/bin/execstack

COPY requirements.txt .
RUN pip install --upgrade pip && \
  pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
  pip install --no-cache-dir -r requirements.txt && \
  # Fix for libctranslate2 executable stack issue
  find /usr/local/lib/python3.11/site-packages/ctranslate2 -name '*.so' -exec execstack -c {} \;

COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs clips audio_files

# Expose the port app runs on (5000)
EXPOSE 5000

# Add healthcheck to monitor container health
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:5000/test-route || exit 1

# Use unbuffered Python output for better logging
ENV PYTHONUNBUFFERED=1

# Run with Python directly for better stability and debugging
CMD ["python", "-u", "app.py"]
