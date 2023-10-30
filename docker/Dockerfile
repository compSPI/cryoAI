FROM fpoitevi/cryoai:latest

WORKDIR /work

ADD requirements.txt /work/requirements.txt

RUN export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX" && \
    pip install torch-points3d

RUN pip install --no-use-pep517 --no-cache-dir -r requirements.txt

ADD . /work