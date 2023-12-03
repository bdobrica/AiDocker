FROM python:3.9-slim-bookworm

### CREATING FOLDERS ###
RUN mkdir -p /opt/app/run
RUN mkdir -p /tmp/ai/staged
RUN mkdir -p /tmp/ai/source
RUN mkdir -p /tmp/ai/prepared

### INSTALLING BASE PYTHON PACKAGES ###
RUN apt-get update && apt-get install -y jq

### MAKE SURE TO INSTALL REQUIREMENTS FIRST, SO THAT THE CACHE CAN BE USED ###
COPY mirnet/requirements.txt /opt/app
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /opt/app/requirements.txt

### COPYING FILES ###
COPY mirnet/container.yaml /opt/app
COPY mirnet/ai /opt/app/ai
COPY daemon /opt/app/daemon
COPY mimetypes.* /opt/app
COPY api /opt/app/api
COPY entry.sh /opt/app

### MAKING THE SCRIPTS EXECUTABLE ###
RUN chmod +x /opt/app/entry.sh

### RUNNING THE SCRIPT ###
ENTRYPOINT ["/opt/app/entry.sh"]
