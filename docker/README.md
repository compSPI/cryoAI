# Container management

## Build and upload to DockerHub

At the moment, we build the cryoAI Docker image in 3 successive steps, building on top of `pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime`:
1. **Step1** CUDA base available on [DockerHub](https://hub.docker.com/repository/docker/fpoitevi/cryonettorch-cuda-base)
2. **Step2** CMAKE base available on [DockerHub](https://hub.docker.com/repository/docker/fpoitevi/cryonettorch-cmake-base)
3. **Step3** Python environment. We provide the Dockerfile for Step3 [here](https://github.com/compSPI/cryoAI/blob/main/docker/Dockerfile).
  We build that image using this [GH worflow](https://github.com/compSPI/cryoAI/blob/main/.github/workflows/build.yml) which is triggered whenever the `Dockerfile` or `requirements.txt` are changed after a push to either the `main` or `docker` branch. available on [Dockerhub](https://hub.docker.com/repository/docker/fpoitevi/cryoai)
  The workflow builds the image and uploads it to [DockerHub](https://hub.docker.com/repository/docker/fpoitevi/cryoai).

This approach provides the benefit of updating the last image only if only the python dependencies change.

Unfortunately we can not provide the Dockerfile for the other steps just yet since they stopped working following an issue similar to the one described here: https://github.com/NVIDIA/nvidia-docker/issues/1631
As soon as a fix has been found we will make the method available.

## Pulling the image

To pull the latest image from DockerHub and convert it to a Singularity image, do the following:
```bash
singularity pull cryoai_latest.sif docker://fpoitevi/cryoai:latest
```

## for Docker experts

For the users interested in helping building the image from scratch, the `Dockerfile` we used successfully to build the CUDA and CMAKE bases linked above are:
- CUDA base:
```docker 
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

MAINTAINER Youssef Nashed "ynashed@slac.stanford.edu"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y wget build-essential

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin && \
    mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub 5 && \
    apt-get update && \
    dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb && \
    apt-get update && \
    apt-get install -y cuda-10-0 && \
    apt-get clean

ENV PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH
```

- CMAKE base:
```docker
FROM fpoitevi/cryonettorch-cuda-base

RUN apt-get update && \
    apt-get install -y libssl-dev

RUN mkdir -p /tmp && cd /tmp \
    && wget https://cmake.org/files/v3.18/cmake-3.18.0.tar.gz \
    && tar xzvf cmake-3.18.0.tar.gz && rm -f cmake-3.18.0.tar.gz \
    && cd cmake-3.18.0 && ./configure && make install \
    && cd /tmp && rm -rf cmake-3.18.0*
```