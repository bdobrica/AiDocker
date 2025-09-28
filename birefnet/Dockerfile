FROM python:3.9-slim-bookworm

### SETTING UP NVIDIA CUDA FOR DEBIAN BOOKWORM ###
ARG NVARCH=x86_64
ARG CUDA_MM=12-6                 # e.g., 12-4, 12-5, 12-6...
ARG CUDA_VERSION=12.6.1          # for ENV + symlink
ARG CUDART_VERSION=12.6.77-1     # apt version string for cuda-cudart-<MM>
ARG CUDA_COMPAT_PKG=cuda-compat-12-6

### SETTINGS FOR AI WORKERS ###
ARG MAX_FORK=1

ENV NVARCH=${NVARCH}
### SIMPLE GUARD TO ENSURE COMPATIBILITY BETWEEN HOST AND CONTAINER DRIVERS ###
ENV NVIDIA_REQUIRE_CUDA="cuda>=${CUDA_VERSION%.*}"

### SETTING UP NVIDIA CUDA FOR DEBIAN BOOKWORM ###
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg2 && \
    curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/debian12/${NVARCH}/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

### INSTALLING NVIDIA CUDA FOR DEBIAN BOOKWORM (RUNTIME ONLY) ###
ENV CUDA_VERSION=${CUDA_VERSION}
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-${CUDA_MM}=${CUDART_VERSION} \
    ${CUDA_COMPAT_PKG} && \
    ln -s /usr/local/cuda-${CUDA_VERSION%.*} /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

### SETTING UP NVIDIA CONTAINER TOOLKIT ###
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
ENV PATH /usr/local/nvidia/bin:${PATH}


### CREATING FOLDERS ###
RUN mkdir -p /opt/app/run
RUN mkdir -p /tmp/ai/staged
RUN mkdir -p /tmp/ai/source
RUN mkdir -p /tmp/ai/prepared

### INSTALLING BASE PYTHON PACKAGES ###
RUN apt-get update && apt-get install -y python3-opencv

### COPYING FILES ###
COPY birefnet/container.yaml /opt/app
COPY birefnet/requirements.txt /opt/app
COPY birefnet/birefnet /opt/app/birefnet
COPY birefnet/ai.py /opt/app
COPY birefnet/birefnet.pth /opt/app
COPY daemon /opt/app/daemon
COPY cleaner.py /opt/app
COPY mimetypes.* /opt/app
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

### WORKERS CONFIGURATION ###
ENV MAX_FORK=${MAX_FORK}

### RUNNING THE SCRIPT ###
ENTRYPOINT ["/opt/app/entry.sh"]
