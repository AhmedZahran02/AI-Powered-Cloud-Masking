# Use official TensorFlow image as base
FROM tensorflow/tensorflow:2.8.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Command to run when container starts
CMD ["python", "train.py"]