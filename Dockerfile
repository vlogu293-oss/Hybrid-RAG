FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY Backend ./Backend
COPY frontend ./frontend
COPY evals.py .

RUN mkdir -p \
    /app/Backend/data \
    /app/Backend/vector_store \
    /app/Backend/evals

EXPOSE 8501

CMD [
    "streamlit",
    "run",
    "frontend/app.py",
    "--server.address=0.0.0.0",
    "--server.port=8501"
]