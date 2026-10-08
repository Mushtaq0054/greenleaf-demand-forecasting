# ==============================================================================
# GreenLeaf Grocery - Perishable Goods Demand Forecasting Microservice
# Task 4: Production Dockerfile
# ==============================================================================

# Use official lightweight Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables for optimized Python execution in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set application working directory inside container
WORKDIR /app

# Install system dependencies if required, then clean apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition first to leverage Docker layer caching
COPY requirements.txt .

# Install Python production dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, model artifact, and necessary modules
COPY . .

# Expose port 8000 for FastAPI service
EXPOSE 8000

# Built-in container health check using Python standard library (no curl package required)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Launch production Uvicorn server bound to 0.0.0.0:8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
