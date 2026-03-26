FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        build-essential \
        ninja-build \
        git \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0

# Optional sanity check during build
RUN which nvcc && nvcc --version

# Upgrade Python packaging tools first
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install the exact CUDA 13.0 vLLM wheel
ARG VLLM_VERSION=0.18.0
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
    "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu130-cp38-abi3-manylinux_2_35_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Remove any conflicting FlashInfer packages first
RUN python3 -m pip uninstall -y flashinfer flashinfer-python flashinfer-cubin flashinfer-jit-cache || true

# Install ONE matched FlashInfer set
RUN python3 -m pip install --no-cache-dir \
    flashinfer-python==0.6.4 \
    flashinfer-cubin==0.6.4 && \
    python3 -m pip install --no-cache-dir \
    flashinfer-jit-cache==0.6.4 \
    --index-url https://flashinfer.ai/whl/cu130

# Install your additional Python dependencies afterwards
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade -r /requirements.txt

# Setup for building the image with the model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4 \
    PYTHONPATH="/:/vllm-workspace"

COPY src /src

RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
      export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
      python3 /src/download_model.py; \
    fi

CMD ["python3", "/src/handler.py"]
