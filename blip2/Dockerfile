FROM python:3.9-slim-bookworm

ENV CUDA_VERSION 12.4.127
ENV CUDA_MINOR_VERSION ${CUDA_VERSION%.*}
ENV NVARCH x86_64
ENV NVIDIA_DRIVER_VERSION 550.54.15
ENV NVIDIA_MAJOR_DRIVER_VERSION ${NVIDIA_DRIVER_VERSION%%.*}
ENV NVIDIA_KEYRING_VERSION 1.1-1

ENV NVARCH x86_64

ENV NVIDIA_REQUIRE_CUDA "cuda>=${CUDA_MINOR_VERSION} brand=nvidiartx,driver>=${NVIDIA_MAJOR_DRIVER_VERSION},driver<$[NVIDIA_MAJOR_DRIVER_VERSION + 1]"
ENV NV_CUDA_CUDART_VERSION "${CUDA_VERSION}"
ENV NV_CUDA_COMPAT_PACKAGE cuda-compat-${CUDA_MINOR_VERSION/./-}
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/debian11/${NVARCH}/cuda-keyring_${NVIDIA_KEYRING_VERSION}_all.deb && \
    dpkg -i cuda-keyring_${NVIDIA_KEYRING_VERSION}_all.deb && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-${CUDA_MINOR_VERSION/./-}=${NV_CUDA_CUDART_VERSION} \
    ${NV_CUDA_COMPAT_PACKAGE} \
    && ln -s cuda-${CUDA_MINOR_VERSION} /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

### CREATING FOLDERS ###
RUN mkdir -p /opt/app/run
RUN mkdir -p /tmp/ai/staged
RUN mkdir -p /tmp/ai/source
RUN mkdir -p /tmp/ai/prepared

### INSTALLING BASE PYTHON PACKAGES ###
RUN apt-get update && apt-get install -y python3-opencv

### COPYING FILES ###
COPY blip2/requirements.txt /opt/app
COPY blip2/config.json /opt/app/blip2/config.json
COPY blip2/merges.txt /opt/app/blip2/merges.txt
COPY blip2/model-00001-of-00002.safetensors /opt/app/blip2/model-00001-of-00002.safetensors
COPY blip2/model-00002-of-00002.safetensors /opt/app/blip2/model-00002-of-00002.safetensors
COPY blip2/model.safetensors.index.json /opt/app/blip2/model.safetensors.index.json
COPY blip2/preprocessor_config.json /opt/app/blip2/preprocessor_config.json
COPY blip2/special_tokens_map.json /opt/app/blip2/special_tokens_map.json
COPY blip2/tokenizer_config.json /opt/app/blip2/tokenizer_config.json
COPY blip2/tokenizer.json /opt/app/blip2/tokenizer.json
COPY blip2/vocab.json /opt/app/blip2/vocab.json
COPY blip2/ai.py /opt/app
COPY daemon /opt/app/daemon
COPY cleaner.py /opt/app
COPY mimetypes.json /opt/app
COPY api.py /opt/app
COPY entry.sh /opt/app

### MAKING THE SCRIPTS EXECUTABLE ###
RUN chmod +x /opt/app/ai.py
RUN chmod +x /opt/app/cleaner.py
RUN chmod +x /opt/app/api.py
RUN chmod +x /opt/app/entry.sh

### INSTALLING OTHER REQUIRED PACKAGES ###
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /opt/app/requirements.txt

### RUNNING THE SCRIPT ###
ENTRYPOINT ["/opt/app/entry.sh"]
