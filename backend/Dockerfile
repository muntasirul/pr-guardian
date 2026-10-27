FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PRGUARDIAN_MOCK=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Mock mode by default (no torch in the image). For real-model serving, build with:
#   pip install -r requirements-ml.txt, set PRGUARDIAN_MOCK=0 and ADAPTER_PATH=/adapters/verdict-v1
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
