ARG PYTHON_VERSION=3.12
FROM python:$PYTHON_VERSION-slim-bookworm
LABEL maintainer="Joris Tavernier <joris.tavernier@openanalytics.eu,>"


## Install missing system dependencies
# also upgrade all packages to avoid vulnerabilities
RUN mkdir -p /usr/share/man/man1 && \
    apt-get --allow-releaseinfo-change-suite update --fix-missing && \
    apt-get upgrade -y && \
    apt-get install libxrender1 libxext6 wget git -y && \
    apt-get install build-essential wkhtmltopdf -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

## nvidia-container runtime compatibility
# copied from https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.7.1/ubuntu2204/base/Dockerfile
# adds approximately 0.13 GiB to the image
ENV CUDA_VERSION=11.7.1
ENV NVIDIA_REQUIRE_CUDA="cuda>=11.7 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471"
ENV NV_CUDA_CUDART_VERSION=11.7.99-1
ENV NV_CUDA_COMPAT_PACKAGE=cuda-compat-11-7
ENV NVARCH=x86_64

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH}/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get purge --autoremove -y curl && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    cuda-cudart-11-7=${NV_CUDA_CUDART_VERSION} \
    ${NV_CUDA_COMPAT_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN pip install --upgrade pip 

## Install dependencies 
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir torch_geometric torchmetrics && \
    pip install --no-cache-dir gpytorch botorch joblib boto3 && \
    pip install --no-cache-dir rdkit feather-format seaborn py3nvml pymongo xgboost tensorboard lightgbm lightning rich ray[tune] matplotlib numpy pandas Cython configargparse pebble tqdm ipython bokeh wandb && \
    pip install --no-cache-dir torchrl tensordict promptsmiles scipy joblib && \
    pip install --no-cache-dir prolif openbabel-wheel && \
    pip install --no-cache-dir ipykernel && \
    python -m ipykernel install && \
    pip install --no-cache-dir ollama langchain_ollama && \
    pip install --no-cache-dir  sqlite-vec pypdfium2 fastembed langchain langchain_experimental sentence-transformers langchain-text-splitters langchain_community langchain langchain_core langchain-openai && \
    pip install --no-cache-dir langgraph langgraph-checkpoint langsmith && \
    pip install --no-cache-dir transformers smolagents[litellm] huggingface_hub[cli] && \
    pip install --no-cache-dir pdfplumber semantic-chunkers open-text-embeddings prompt-template

RUN pip install --no-cache-dir  gradio  pyprojroot python-dotenv PyYAML SQLAlchemy openai pillow && \
    pip install --no-cache-dir  "praisonaiagents[knowledge]"

RUN pip install --no-cache-dir --upgrade jupyterlab jupyterlab-server jupyter_core jupyter_client jupyter-server
RUN pip install --no-cache-dir ipywidgets ipython_genutils pykan ticc==0.1.4 oracledb datamol safe-mol mdtraj pdb2pqr MDAnalysis py3Dmol
