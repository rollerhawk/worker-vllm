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

RUN which nvcc && nvcc --version

RUN python3 -m pip install --upgrade pip setuptools wheel

# Install vLLM — this pulls compatible torch + flashinfer-python automatically
ARG VLLM_VERSION=0.18.0
RUN python3 -m pip install --no-cache-dir \
    "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu130-cp38-abi3-manylinux_2_35_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Pin torch and capture flashinfer version so nothing downstream can break them
RUN python3 -c "\
import torch, flashinfer; \
open('/torch-constraint.txt','w').write(f'torch=={torch.__version__}\n'); \
open('/flashinfer-version.txt','w').write(flashinfer.__version__)" && \
    cat /torch-constraint.txt && \
    echo "flashinfer-python=$(cat /flashinfer-version.txt)"

# Install matching cubin + jit-cache for the flashinfer version vLLM chose
RUN FI_VERSION=$(cat /flashinfer-version.txt) && \
    python3 -m pip install --no-cache-dir \
    flashinfer-cubin==${FI_VERSION} \
    -c /torch-constraint.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 && \
    python3 -m pip install --no-cache-dir \
    flashinfer-jit-cache==${FI_VERSION} \
    --index-url https://flashinfer.ai/whl/cu130 \
    -c /torch-constraint.txt

# Install additional Python dependencies — constrained
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade -r /requirements.txt \
    -c /torch-constraint.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Verify everything is consistent
RUN python3 -c "\
import torch, vllm, flashinfer; \
print(f'torch {torch.__version__} cuda {torch.version.cuda}'); \
print(f'vllm {vllm.__version__}'); \
print(f'flashinfer {flashinfer.__version__}')"

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
