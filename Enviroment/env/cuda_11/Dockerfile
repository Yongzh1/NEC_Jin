# Base image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime as pytorch
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as nvidia

# Environment variables
ENV CUDA_MAJOR_VERSION=11
ENV CUDA_MINOR_VERSION=6
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
    python3-pip \
    python3-setuptools \
    python3-distutils && \
    apt-get install -y libfreetype6-dev && \
    apt-get install -y libglib2.0-0 libxext6 libsm6 libxrender1 libfontconfig1 --fix-missing && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Additional Argument variable
ARG CONDA_DIR=/opt/CONDA_DIR
# add it to path
ENV PATH ${CONDA_DIR}/bin:${PATH}
# pipのアップグレード

RUN pip install --upgrade pip
# Pip install packages
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# RUN pip install --upgrade cython && \
#    pip install --upgrade cysignals && 
RUN pip install --no-cache-dir \
    albumentations \
    astroid \
    attrs \
    bokeh \
    boto3 \
    defusedxml \
    docker-pycreds \
    docutils \
    fastai \
    future \
    gast \
    gensim \
    gitdb \
    glibc \
    GitPython \
    google-auth \
    google-auth-oauthlib==0.4.1 \
    grpcio \
    isort \
    isoweek \
    Jinja2 \
    jmespath \
    joblib \
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
    parso \
    plotly \
    Pillow \
    pluggy \
    pytest \
    pydot \
    python-dateutil \
    pytz \
    pyarrow \
    regex \
    requests \
    requests-oauthlib \
    retry \
    s3transfer \
    scikit-image \
    scikit-learn \
    scikit-learn-intelex \
    scipy==1.8.0 \
    seaborn \
    segtok \
    srsly \
    statsmodels \
    stevedore \
    termcolor 
RUN pip install --no-cache-dir \
   # tokenizers==0.11.1 \
    toolz \
    tornado \
    tqdm \
    textblob \
    traitlets \
    tabulate \
    transformers \
    urllib3 \
    wcwidth \
    webencodings \
    zict \
    zipp \
    joblib 


# Expose the port to the outside Docker (in your broswer)
EXPOSE 8888

# Automatically run these command in the end
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
