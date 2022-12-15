#!/bin/bash

find . -name "Dockerfile" | while read dockerfile; do
    model_dir=$(dirname ${dockerfile})
    model_name=$(basename ${model_dir})
    weights_file="${model_dir}/weights.txt"

    echo "Building ${model_name} ..."

    if [ -f "${weights_file}" ]; then
        cat "${weights_file}" | while read line; do
            line=$(echo ${line} | tr -d "\r" | tr -d "\n")
            if [ ! -z "${line}" ]; then
                weight_file=${model_dir}/${line#*/}
                if [ ! -f "${weight_file}" ]; then
                    wget "https://ublo.ro/wp-content/mirror/${line}" -O ${weight_file}
                fi
            fi
        done
    fi
    docker build -f "${dockerfile}" -t "${model_name}" .
done
