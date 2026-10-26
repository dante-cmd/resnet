FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_CHECKPOINT=/model/best.pt

WORKDIR /app

# CPU-only torch keeps the image small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY serving ./serving
RUN pip install --no-cache-dir .

# Bake the trained checkpoint in at build time:
#   docker build -f serving/Dockerfile -t resnet-mlops --build-arg CHECKPOINT=checkpoints/best.pt .
ARG CHECKPOINT=checkpoints/best.pt
COPY ${CHECKPOINT} /model/best.pt

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
