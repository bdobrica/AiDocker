#!/bin/bash

# --events-backend=file
debug_mode=false
container=""
docker_args=()
env_file="./docker.env"

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

function detect_cuda {
    if command -v nvidia-smi &> /dev/null; then
        echo "nvidia-smi found, checking free GPU memory ..."
        free_gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -n 1)
        if [ "${free_gpu}" -lt 1000 ]; then
            echo "No free GPU memory found, using CPU"
            return
        fi
        echo "Free GPU memory found, using GPU"
        docker_args+=("--gpus" "all")
    else
        echo "nvidia-smi not found, using CPU"
        return
    fi
}

function get_next_port {
    printf "%s\n%s\n%s\n" "$(\
        find . -name port.txt | while read f; do \
            printf "%05d\n" $(cat $f); done \
        | sort | tail -n 1 \
    )" "$(\
        find . -name container.yaml | while read f; do \
            printf "%05d\n" $(cat $f | yq -r '.port'); done \
        | sort | tail -n 1 \
    )" "05000" | sort | tail -n 1 | awk "{print $1 + 1}"
}

function run_container {
    local dockerfile="$1"
    shift
    local container_args=("$@")
    local model_dir=$(dirname "${dockerfile}")
    local model_name=$(basename "${model_dir}")
    local docker=$(detect_docker_command)
    local config_file="${model_dir}/container.yaml"
    local port=""

    if [ -f "${config_file}" ]; then
        echo "Found container.yaml, searching for port ..."
        port=$(cat "${config_file}" | yq -r '.port')
    elif [ -f "${model_dir}/port.txt" ]; then
        echo "Found port.txt, searching for port ..."
        port=$(cat "${model_dir}/port.txt" | tr -d "\r" | tr -d "\n")
    fi

    if [ -z "${port}" ]; then
        echo "No port found, searching for next available port ..."
        port=$(get_next_port)
        echo "Found port ${port}. Saving to port.txt ..."
        echo "${port}" > "${model_dir}/port.txt"
    fi

    docker_args+=(-p 127.0.0.1:${port}:5000/tcp)

    if [ -f "${env_file}" ]; then
        echo "Container environment variables:"
        cat "${env_file}" | while read line; do
            echo " >> ${line}"
        done
    fi

    docker_args+=(--rm)
    docker_args+=(--env-file "${env_file}")

    if [ "${debug_mode}" = true ]; then
        echo "Running ${model_name} on port ${port} in debug mode ..."
        $docker run \
            ${docker_args[@]} \
            -e DEBUG=true \
            -it \
            --entrypoint /bin/bash \
            ${model_name}
    else
        echo "Running ${model_name} on port ${port} ..."
        $docker run \
            ${docker_args[@]} \
            -d \
            ${model_name} ${container_args[@]}
    fi
}

function show_usage {
    echo "Usage: $0 [-h] [-c <container>] [-a] [-d]"
    echo "  -h  Show this help message"
    echo "  -c  Run only the specified container"
    echo "  -a  Run all containers"
    echo "  -d  Enable debug mode"
    echo "  -n  Run connected to this network"
    echo "  -e  Run with this environment file"
    echo "  -v  Run with this volume mounted under /opt/app, read-only"
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
            d)
                echo "Debug mode enabled"
                debug_mode=true
                ;;
            n)
                echo "Connecting to network ${OPTARG}"
                docker_args+=("--network" "${OPTARG}")
                ;;
            e)
                echo "Using environment file ${OPTARG}"
                env_file="${OPTARG}"
                if [ ! -f "${env_file}" ]; then
                    echo "Environment file ${env_file} does not exist"
                    exit 1
                fi
                ;;
            v)
                echo "Using volume ${OPTARG} for /opt/app"
                docker_args+=("-v" "$(realpath ${OPTARG}):/opt/app:ro")
                ;;
            *)
                show_usage
                ;;
        esac
    done
}

parse_arguments "$@"
detect_cuda

if [ -z "${container}" ]; then
    echo "No container specified"
    show_usage
    exit 1
fi

if [ "${container}" != "all" ]; then
    run_container "${container}/Dockerfile"
else
    if [ "${debug_mode}" = true ]; then
        echo "Debug mode is not supported when starting all containers"
        exit 1
    fi
    echo "Running all containers ..."
    find . -name "Dockerfile" | while read dockerfile; do
        run_container "${dockerfile}"
    done
fi
