FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# 必要パッケージ
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# conda環境構築
COPY environment.yml .

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda env create -f environment.yml && \
    conda clean -afy

# PATH設定
ENV PATH /opt/conda/envs/mol-regression-env/bin:$PATH

# 自動でactivateされるように
RUN echo "conda activate mol-regression-env" >> ~/.bashrc
