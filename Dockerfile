# Use the official NVIDIA PyTorch image with optimized libraries
# This version includes CUDA 12.1 and cuDNN pre-installed
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set the working directory
WORKDIR /workspace

# Install system dependencies for Zarr and FFT operations
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    fftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# We install xformers specifically for memory-efficient attention
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir xformers==0.0.22.post7

# Copy the entire project
COPY . .

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TORCH_CUDNN_V8_API_ENABLED=1

# Expose port for experiment tracking (e.g., TensorBoard or WandB)
EXPOSE 6006

# The default command runs the training pipeline
# Can be overridden to run inference or export
CMD ["python", "main.py"]
