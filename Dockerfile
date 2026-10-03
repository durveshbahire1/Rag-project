# Dockerfile for the FastAPI backend.
# Build with:  docker build -t rag-api .
# Run with:    docker run -p 8000:8000 --env-file .env rag-api

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
