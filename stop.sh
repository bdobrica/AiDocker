#!/bin/bash

find . -type f -name "Dockerfile" | while read dockerfile; do
  model_dir=$(dirname "${dockerfile}")
  model_name=$(basename "${model_dir}")
  docker ps | grep "${model_name}" | awk '{print $1}' | xargs docker stop > /dev/null 2>&1
done