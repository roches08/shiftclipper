FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.runpod.txt /app/requirements.runpod.txt
RUN pip install --no-cache-dir -r /app/requirements.runpod.txt
COPY . /app

ENV PYTHONUNBUFFERED=1
