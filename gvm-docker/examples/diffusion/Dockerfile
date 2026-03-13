FROM nvidia/cuda:12.0.0-base-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA runtime libraries (needed for PyTorch)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-cudart-12-0 cuda-compat-12-0 && \
    rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

# Create app directory
WORKDIR /app

# Create virtual environment and install packages
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install PyTorch and diffusion dependencies
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir \
    diffusers \
    transformers \
    protobuf \
    sentencepiece \
    accelerate

# Download diffusion script and prompts
RUN wget https://github.com/ovg-project/GVM/releases/download/v0.0.0-diffusion/diffusion.py && \
    wget https://github.com/ovg-project/GVM/releases/download/v0.0.0-diffusion/vidprom.txt

# Create output directory
RUN mkdir -p /app/outputs

# Set GVM intercept layer (will be mounted from host)
ENV LD_LIBRARY_PATH=/gvm-cuda-driver/install:$LD_LIBRARY_PATH

# Default command
CMD ["python3", "diffusion.py", "--dataset_path=vidprom.txt", "--num_requests=10", "--log_file=outputs/stats.txt"]
