#!/bin/bash

max_port=5000
find . -type f -name "Dockerfile" | while read dockerfile; do
  model_dir=$(dirname "${dockerfile}")
  model_name=$(basename "${model_dir}")

  if [ -f "${model_dir}/port.txt" ]; then
    port=$(cat "${model_dir}/port.txt")
    if [ -e "${port}" ]; then
        port=$[max_port + 1]
        echo "${port}" > "${model_dir}/port.txt"
    fi
  else
    port=$[max_port + 1]
    echo "${port}" > "${model_dir}/port.txt"
  fi

  if [ "${port}" -gt "${max_port}" ]; then
    max_port="${port}"
  fi

  docker run --rm --env-file ./docker.env -d -p 127.0.0.1:${port}:5000/tcp ${model_name}
done