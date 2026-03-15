FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install CPU-only PyTorch first
RUN pip install torch==2.3.0+cpu torchvision==0.18.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install flask gunicorn numpy Pillow \
    scikit-learn xgboost timm==0.9.12

# Copy project files
COPY application.py .
COPY models/ ./models/
COPY templates/ ./templates/
COPY static/ ./static/

# Expose port
EXPOSE 8000

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "120", "--workers", "2", "application:application"]
