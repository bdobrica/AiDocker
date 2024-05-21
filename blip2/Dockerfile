FROM python:3.9-slim-bookworm

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
