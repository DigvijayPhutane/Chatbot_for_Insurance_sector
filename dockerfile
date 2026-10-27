FROM python:3.12-slim

WORKDIR /app

# CPU-only torch: this machine has no NVIDIA GPU, and the CUDA build adds several GB.
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
