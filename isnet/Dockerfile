FROM python:3.9-slim-bullseye

### CREATING FOLDERS ###
RUN mkdir -p /opt/app/run
RUN mkdir -p /tmp/ai/staged
RUN mkdir -p /tmp/ai/source
RUN mkdir -p /tmp/ai/prepared

### INSTALLING BASE PYTHON PACKAGES ###
RUN apt-get update && apt-get install -y python3-opencv

### COPYING FILES ###
COPY isnet/container.yaml /opt/app
COPY isnet/requirements.txt /opt/app
COPY isnet/isnet /opt/app/isnet
COPY isnet/ai.py /opt/app
COPY isnet/isnet-general-use.pth /opt/app
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