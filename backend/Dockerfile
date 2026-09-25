FROM python:3.11-slim

WORKDIR /workspace

# Native libraries needed for OpenSlide runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.base.txt
COPY backend/requirements.txt /workspace/backend/requirements.txt

RUN pip install --no-cache-dir \
    -r /workspace/requirements.base.txt \
    -r /workspace/backend/requirements.txt

# CPU PyTorch runtime for the inference worker. Developers with CUDA can use
# the matching PyTorch build from the official selector instead.
RUN pip install --no-cache-dir torch torchvision

COPY . /workspace

ENV PYTHONPATH=/workspace
ENV PATHOVISION_API_HOST=0.0.0.0
ENV PATHOVISION_API_PORT=8000

EXPOSE 8000

CMD [
  "uvicorn",
  "backend.app.main:app",
  "--host",
  "0.0.0.0",
  "--port",
  "8000"
]
