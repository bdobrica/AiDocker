### USE A BUILDER STAGE TO BUILD DLIB ###
FROM python:3.9-slim-bullseye as builder

RUN apt-get update && apt-get install -y build-essential cmake
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install dlib
WORKDIR /usr/local/lib/python3.9/site-packages/
RUN find . -name "*dlib*" -exec tar -rf dlib.tar {} \;

FROM python:3.9-slim-bullseye

### COPY DLIB ###
WORKDIR /usr/local/lib/python3.9/site-packages/
COPY --from=builder /usr/local/lib/python3.9/site-packages/dlib.tar .
RUN tar -xf dlib.tar
WORKDIR /

### CREATING FOLDERS ###
RUN mkdir -p /opt/app/run
RUN mkdir -p /tmp/ai/staged
RUN mkdir -p /tmp/ai/source
RUN mkdir -p /tmp/ai/prepared

### INSTALLING BASE PYTHON PACKAGES ###
RUN apt-get update && apt-get install -y python3-opencv

### COPYING FILES ###
COPY agenet/requirements.txt /opt/app
COPY agenet/container.yaml /opt/app
COPY agenet/EfficientNetB3_224_weights.11-3.44.hdf5 /opt/app
COPY agenet/efficientnetb3_notop.h5 /opt/app
COPY agenet/ai.py /opt/app
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

### RUNNING THE SCRIPT ###
ENTRYPOINT ["/opt/app/entry.sh"]
