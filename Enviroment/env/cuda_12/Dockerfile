# Base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 as nvidia

# Environment variables
ENV CUDA_MAJOR_VERSION=12
ENV CUDA_MINOR_VERSION=2
ENV CUDA_VERSION=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/bin:$PATH

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION"

# Linux commands to update and install base packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    unzip \
    cmake \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    p7zip-full \
    openssh-client \
    python3.9 \
    python3.9-venv \
    python3.9-distutils \
    libglib2.0-0 libxext6 libsm6 libxrender1 libfontconfig1 --fix-missing && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Additional Argument variable
ARG CONDA_DIR=/opt/CONDA_DIR
# add it to path
ENV PATH ${CONDA_DIR}/bin:${PATH}

# Create a symbolic link for python3.9 as python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Create virtual environment
RUN python -m venv /venv

# Upgrade pip and install packages
RUN /venv/bin/pip install --upgrade pip
RUN /venv/bin/pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nvidia-cuda-runtime-cu12

# Install additional Python packages
RUN /venv/bin/pip install --no-cache-dir \
    defusedxml \
    docker-pycreds \
    docutils \
    gensim \
    GitPython \
    json5 \
    jupyter \
    jupyterlab \
    jupyter-client \
    jupyter-console \
    jupyter-core \
    jupyterlab-lsp \
    papermill \
    numpy \
    oauthlib \
    opencv-python \
    optuna \
    pandas \
    pandocfilters \
    plotly \
    Pillow \
    pydot \
    python-dateutil \
    scikit-image \
    scikit-learn \
    scipy \
    statsmodels \
    stevedore \
    termcolor \
    tqdm \
    textblob \
    transformers \
    joblib

# Expose the port to the outside Docker (in your browser)
EXPOSE 8892

# Automatically run these commands in the end
ENTRYPOINT ["/venv/bin/jupyter", "lab", "--ip=0.0.0.0", "--port=8892", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
