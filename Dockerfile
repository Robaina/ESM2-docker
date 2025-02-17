# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference script
COPY inference.py .

# Create input and output directories
RUN mkdir /app/input /app/output

# Default command (can be overridden)
ENTRYPOINT ["python", "inference.py"]