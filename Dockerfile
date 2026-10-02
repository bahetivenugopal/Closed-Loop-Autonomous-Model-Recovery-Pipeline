FROM python:3.10-slim

WORKDIR /app

# Install system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps separately (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY src ./src
COPY scripts ./scripts
COPY data/ ./data/

# Create dirs
RUN mkdir -p /app/data /app/models /mlflow

EXPOSE 8000
CMD ["uvicorn", "src.api_mlflow:app", "--host", "0.0.0.0", "--port", "8000"]