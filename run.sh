#!/bin/bash

# --events-backend=file
max_port=5000
debug_mode=false
container=""
docker_args=()

function detect_docker_command {
    if [ -x "$(command -v docker)" ]; then
        echo "sudo docker"
    elif [ -x "$(command -v podman)" ]; then
        echo "podman"
    else
        echo "sudo docker"
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
    local model_dir=$(dirname "${dockerfile}")
    local model_name=$(basename "${model_dir}")
    local docker=$(detect_docker_command)

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

    if [ "${debug_mode}" = true ]; then
        echo "Running ${model_name} on port ${port} in debug mode ..."
        $docker run ${docker_args[@]} --rm --env-file ./docker.env -it --entrypoint /bin/bash -p 127.0.0.1:${port}:5000/tcp ${model_name}
    else
        echo "Running ${model_name} on port ${port} ..."
        $docker run ${docker_args[@]} --rm --env-file ./docker.env -d -p 127.0.0.1:${port}:5000/tcp ${model_name}
    fi
}

function show_usage {
    echo "Usage: $0 [-h] [-c <container>] [-a] [-d]"
    echo "  -h  Show this help message"
    echo "  -c  Run only the specified container"
    echo "  -a  Run all containers"
    echo "  -d  Enable debug mode"
    exit 0
}

function parse_arguments {
    while getopts ":hc:ad" opt; do
        case "${opt}" in
            h)

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