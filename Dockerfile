# Base image with PyTorch and CUDA
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, GDAL (rasterio), and other libs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgdal-dev \
    python3-gdal \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (excluding files in .dockerignore)
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default command (override when running)
CMD ["python", "run_inference.py", \
    "--test_folder", "/app/test", \
    "--model_path", "/app/outputs/models/Unet_model.pth", \
    "--output_csv", "/app/outputs/predictions/submission.csv", \
    "--threshold", "0.35"]