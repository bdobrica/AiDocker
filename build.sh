#!/bin/bash

function build_container {
    local dockerfile="$1"
    local model_dir=$(dirname ${dockerfile})
    local model_name=$(basename ${model_dir})
    local weights_file="${model_dir}/weights.txt"

    echo "Building ${model_name} ..."

    if [ -f "${weights_file}" ]; then
        cat "${weights_file}" | while read line; do
            line=$(echo ${line} | tr -d "\r" | tr -d "\n")
            if [ ! -z "${line}" ]; then
                local weight_file=${model_dir}/${line#*/}
                if [ ! -f "${weight_file}" ]; then
                    wget "https://ublo.ro/wp-content/mirror/${line}" -O ${weight_file}
                fi
            fi
        done
    fi
    docker build -f "${dockerfile}" -t "${model_name}" .
}


function parse_arguments {
    while getopts ":hc:a" opt "$@"; do
        case "${op}t" in
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
