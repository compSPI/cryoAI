FROM fpoitevi/cryoai:latest

WORKDIR /work

ADD requirements.txt /work/requirements.txt

RUN pip install torch-points3d

RUN pip install --no-use-pep517 --no-cache-dir -r requirements.txt

ADD . /work