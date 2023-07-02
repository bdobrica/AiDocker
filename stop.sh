#!/bin/bash

function detect_docker_command {
    if pgrep dockerd >/dev/null && [ -x "$(command -v docker)" ]; then
        echo "sudo docker"
    elif [ -x "$(command -v podman)" ]; then
        echo "podman"
    else
        echo "No docker command found"
        exit 1
    fi
}

docker=$(detect_docker_command)

find . -type f -name "Dockerfile" | while read dockerfile; do
  model_dir=$(dirname "${dockerfile}")
  model_name=$(basename "${model_dir}")
  $docker | grep "${model_name}" | awk '{print $1}' | xargs $docker stop > /dev/null 2>&1
done
