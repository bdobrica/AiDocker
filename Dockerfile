FROM python:3.7-slim-bullseye

### CREATING FOLDERS ###
RUN mkdir -p /opt/app/run
RUN mkdir -p /tmp/ai/staged
RUN mkdir -p /tmp/ai/source
RUN mkdir -p /tmp/ai/prepared

### INSTALLING BASE PYTHON PACKAGES ###
RUN apt-get update && apt-get install -y python3-opencv

### COPYING FILES ###
COPY requirements.txt /opt/app
COPY modnet_photographic_portrait_matting.onnx /opt/app
COPY daemon /opt/app/daemon
COPY ai.py /opt/app
COPY api.py /opt/app
COPY entry.sh /opt/app

### MAKING THE SCRIPTS EXECUTABLE ###
RUN chmod +x /opt/app/ai.py
RUN chmod +x /opt/app/api.py
RUN chmod +x /opt/app/entry.sh

### INSTALLING OTHER REQUIRED PACKAGES ###
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /opt/app/requirements.txt

### RUNNING THE SCRIPT ###
ENTRYPOINT ["/opt/app/entry.sh"]