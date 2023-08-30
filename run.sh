#!/bin/bash

# --events-backend=file
max_port=5000
debug_mode=false
container=""
docker_args=()
network=""
env_file="./docker.env"

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

function run_container {
    local dockerfile="$1"
    shift
    local container_args=("$@")
    local model_dir=$(dirname "${dockerfile}")
    local model_name=$(basename "${model_dir}")
    local docker=$(detect_docker_command)

    if [ -f "${model_dir}/port.txt" ]; then
        port=$(cat "${model_dir}/port.txt" | tr -d "\r" | tr -d "\n")
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

    if [ -n "${network}" ]; then
        if ! $docker network inspect "${network}" >/dev/null; then
            echo "Network ${network} does not exist"
            exit 1
        fi
        docker_args+=("--network" "${network}")
    fi
    docker_args+=(-p 127.0.0.1:${port}:5000/tcp)

    if [ -f "${env_file}" ]; then
        echo "Container environment variables:"
        cat "${env_file}" | while read line; do
            echo " >> ${line}"
        done
    fi

    if [ "${debug_mode}" = true ]; then
        echo "Running ${model_name} on port ${port} in debug mode ..."
        $docker run \
            ${docker_args[@]} \
            --rm \
            --env-file $env_file \
            -e DEBUG=true \
            -it \
            --entrypoint /bin/bash \
            ${model_name}
    else
        echo "Running ${model_name} on port ${port} ..."
        $docker run \
            ${docker_args[@]} \
            --rm \
            --env-file $env_file \
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
    exit 0
}

function parse_arguments {
    while getopts ":hc:adn:e:" opt; do
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
                network="${OPTARG}"
                ;;
            e)
                echo "Using environment file ${OPTARG}"
                env_file="${OPTARG}"
                if [ ! -f "${env_file}" ]; then
                    echo "Environment file ${env_file} does not exist"
                    exit 1
                fi
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
        echo "Debug mode is not supported for all containers"
        exit 1
    fi
    echo "Running all containers ..."
    find . -name "Dockerfile" | while read dockerfile; do
        run_container "${dockerfile}"
    done
fi
