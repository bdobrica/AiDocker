#!/bin/bash

find . -name "Dockerfile" | while read dockerfile; do
    model_dir=$(dirname ${dockerfile})
    model_name=$(basename ${model_dir})
    WEIGHTS_FILE="${model_dir}/weights.txt"
    if [ -f "${WEIGHTS_FILE}" ]; then
        cat "${WEIGHTS_FILE}" | while read line; do
            line=$(echo ${line} | tr -d "\r" | tr -d "\n")
            if [ ! -z "${line}" ]; then
                weight_file=${model_dir}/$(basename ${line})
                if [ ! -f "${weight_file}" ]; then
                    wget "https://ublo.ro/wp-content/mirror/${line}" -O ${weight_file}
                fi
            fi
        done
    fi
    docker build -f "${dockerfile}" -t "${model_name}" .
done
