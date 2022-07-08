FROM fpoitevi/cryoai-cuda-base

RUN apt-get update && \
    apt-get install -y libssl-dev

RUN mkdir -p /tmp && cd /tmp \
    && wget https://cmake.org/files/v3.18/cmake-3.18.0.tar.gz \
    && tar xzvf cmake-3.18.0.tar.gz && rm -f cmake-3.18.0.tar.gz \
    && cd cmake-3.18.0 && ./configure && make install \
    && cd /tmp && rm -rf cmake-3.18.0*