FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# 必要パッケージ
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# ★ build context は repo root なので、COPY は env/... を直接参照できる
COPY env/environment_train_linux_gpu.yml /tmp/environment.yml

# conda環境構築
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda env create -f /tmp/environment.yml && \
    conda clean -afy

# PATH設定（environment.yml の name: mol-regression-env と一致している前提）
ENV PATH /opt/conda/envs/mol-regression-env/bin:$PATH

# 自動でactivateされるように
RUN echo "conda activate mol-regression-env" >> ~/.bashrc
