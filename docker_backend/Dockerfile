FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY docker_requirements.txt .
RUN pip install --no-cache-dir -r docker_requirements.txt
RUN pip install --no-cache-dir --no-binary=xgboost xgboost==3.0.2

COPY api/ /app/api/
COPY inference/ /app/inference/
COPY utils/ /app/utils/
COPY artifacts/ /app/artifacts/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

