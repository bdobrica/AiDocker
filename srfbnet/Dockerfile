FROM python:3.12-slim-bookworm

### SETTING UP NVIDIA CUDA FOR DEBIAN BOOKWORM ###
# Check: https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/index.html
ARG NVARCH=x86_64
ARG CUDA_MM=13-0                 # e.g., 12-4, 12-5, 12-6...
ARG CUDA_VERSION=13.0.88         # for ENV + symlink
ARG CUDART_VERSION=13.0.88-1     # apt version string for cuda-cudart-<MM>
ARG CUDA_COMPAT_PKG=cuda-compat-13-0

### SETTINGS FOR AI WORKERS ###
ARG MAX_FORK=1
ARG QUEUE_LATENCY=0.1

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
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    jq build-essential python3-dev \
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

### INSTALLING OTHER REQUIRED PACKAGES ###
COPY srfbnet/requirements.txt /opt/app
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /opt/app/requirements.txt

### COPYING FILES ###
COPY srfbnet/container.yaml /opt/app
COPY srfbnet/srfbnet /opt/app/srfbnet
COPY srfbnet/ai.py /opt/app
COPY srfbnet/gmfn_x2.pth /opt/app
COPY srfbnet/gmfn_x3.pth /opt/app
COPY srfbnet/gmfn_x4.pth /opt/app
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

### WORKERS CONFIGURATION ###
ENV MAX_FORK ${MAX_FORK}
ENV QUEUE_LATENCY ${QUEUE_LATENCY}

### RUNNING THE SCRIPT ###
ENTRYPOINT ["/opt/app/entry.sh"]
