FROM python:3.9-slim-bookworm

### CREATING FOLDERS ###
RUN mkdir -p /opt/app

### INSTALLING BASE PYTHON PACKAGES ###
RUN apt-get update && apt-get install -y jq

### MAKE SURE TO INSTALL REQUIREMENTS FIRST, SO THAT THE CACHE CAN BE USED ###
COPY webui/requirements.txt /opt/app
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /opt/app/requirements.txt

### COPYING FILES ###
COPY webui /opt/app/webui

WORKDIR /opt/app

### RUNNING THE SCRIPT ###
ENTRYPOINT ["python3", "-m", "webui"]
