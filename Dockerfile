FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/app/nltk_data

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py retreival.py llm.py database.py ./

# Setup permission
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/temp_uploads /app/nltk_data \
    && chown -R appuser:appuser /app
USER appuser

# Pre-download NLTK dictionaries so the app boots instantly
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/app/nltk_data'); nltk.download('stopwords', download_dir='/app/nltk_data')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]