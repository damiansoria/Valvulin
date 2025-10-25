# Dockerfile for running the Valvulin trading bot
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV VALVULIN_CONFIG=/app/config.yaml

CMD ["python", "-m", "core.engine", "--config", "/app/config.yaml"]
