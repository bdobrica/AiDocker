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

function stop_container {
    local container="$1"
    local docker=$(detect_docker_command)
    echo "Stopping ${container} ..."
    $docker ps | grep "${container}" | awk '{print $1}' | xargs $docker stop > /dev/null 2>&1
}

function show_usage {
    echo "Usage: $0 [-h] [-c <container>] [-a] [-d]"
    echo "  -h  Show this help message"
    echo "  -c  Stop only the specified container"
    echo "  -a  Stop all containers"
    exit 0
}

function parse_arguments {
    while getopts ":hc:adn:e:v:" opt; do
        case "${opt}" in
            h)
                show_usage
                ;;
            c)
                if [ -f "${OPTARG}/Dockerfile" ]; then
                    container="${OPTARG}"
                else
                    echo "Container ${OPTARG} does not exist"
                    exit 1
                fi
                ;;
            a)
                container="all"
                ;;
            *)
                show_usage
                ;;
        esac
    done
}

parse_arguments "$@"

if [ -z "${container}" ]; then
    echo "No container specified"
    show_usage
    exit 1
fi

if [ "${container}" != "all" ]; then
    stop_container "${container}"
    exit 0
else
    find . -type f -name "Dockerfile" | while read dockerfile; do
        model_dir=$(dirname "${dockerfile}")
        model_name=$(basename "${model_dir}")
        stop_container "${model_name}"
    done
fi
