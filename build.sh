#!/bin/bash

function yq { python3 -c "import json,sys,yaml;print(json.dumps(yaml.safe_load(sys.stdin)))" | jq "$@"; }

function detect_docker_command {
    if pgrep dockerd >/dev/null && [ -x "$(command -v docker)" ]; then
        if [ "${EUID}" -ne 0 ]; then
            echo "sudo docker"
        else
            echo "docker"
        fi
    elif [ -x "$(command -v podman)" ]; then
        echo "podman"
    else
        echo "No docker command found"
        exit 1
    fi
}

function build_container {
    local dockerfile="$1"
    local model_dir=$(dirname ${dockerfile})
    local model_name=$(basename ${model_dir})
    local weights_file="${model_dir}/weights.txt"
    local config_file="${model_dir}/container.yaml"
    local docker=$(detect_docker_command)
    local weights=()

    echo "Building ${model_name} ..."

    if [ -f "${config_file}" ]; then
        echo "Found container.yaml, searching for weights ..."
        weights=( $(cat "${config_file}" | yq -r '.weights[]') ) 
    elif [ -f "${weights_file}" ]; then
        echo "Found weights.txt, searching for weights ..."
        weights=( $(cat "${weights_file}" | tr -d "\r" | tr -d "\n") )
    fi

    if [ "${#weights[@]}" -gt 0 ]; then
        echo "Downloading model weights ..."
        for weight in "${weights[@]}"; do
            if [ -n "${weight}" ]; then
                local weight_file=${model_dir}/${weight#*/}
                if [ ! -f "${weight_file}" ]; then
                    echo "Creating directory $(dirname ${weight_file})"
                    mkdir -p $(dirname ${weight_file})
                    echo "Downloading ${line} to ${weight_file} ..."
                    wget "https://ublo.ro/wp-content/mirror/${line}" -O ${weight_file}
                else
                    echo "Found ${weight_file}, skipping download"
                fi
            fi
        done
    else
        echo "No weights found, skipping weights download"
    fi
    
    $docker build -f "${dockerfile}" -t "${model_name}" .
}


function parse_arguments {
    while getopts ":hc:a" opt; do
        case "${opt}" in
            h)
                echo "Usage: $0 [-h] [-c <container>] [-a]"
                echo "  -h  Show this help message"
                echo "  -c  Build only the specified container"
                echo "  -a  Build all containers"
                exit 0
                ;;
            c)
                if [ -f "${OPTARG}/Dockerfile" ]; then
                    build_container "${OPTARG}/Dockerfile"
                    exit 0
                fi
                build_container "${OPTARG}"
                exit 0
                ;;
            a)
                echo "Building all containers ..."
                find . -name "Dockerfile" | while read dockerfile; do
                    build_container "${dockerfile}"
                done
                ;;
            *)
                echo "Invalid option: -${OPTARG}"
                exit 1
                ;;
        esac
    done
}

parse_arguments "$@"
