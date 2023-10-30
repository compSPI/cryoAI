FROM fpoitevi/cryoai:latest

WORKDIR /work

ADD requirements.txt /work/requirements.txt

RUN export CUDA_PATH=/usr/local/nvidia/cuda/11.1 && \
    export TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" && \
    pip install torch-points3d

RUN pip install --no-use-pep517 --no-cache-dir -r requirements.txt

ADD . /work