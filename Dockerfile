FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py train.py train_classification.py dataset_source.py Procfile ./

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV DATASET_ZIP_URL=https://drive.google.com/file/d/1xJHs0Jsy6pXJOsv_RApupq8X49JeiPM9/view?usp=sharing

EXPOSE 8080

# Shell form so $PORT expands (Railway Dockerfile startCommand uses exec form otherwise)
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 120"]
