FROM fpoitevi/cryoai:latest

WORKDIR /work

ADD requirements.txt /work/requirements.txt

ENV TORCH_CUDA_ARCH_LIST "8.0;8.6+PTX"

RUN pip install torch-points3d

RUN pip install --no-use-pep517 --no-cache-dir -r requirements.txt

ADD . /work