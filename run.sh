#!/bin/bash

# --events-backend=file
max_port=5000
debug_mode=false

function run_container {
    local dockerfile="$1"
    local model_dir=$(dirname "${dockerfile}")
    local model_name=$(basename "${model_dir}")

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
        sudo docker run --rm --env-file ./docker.env -it --entrypoint /bin/bash -p 127.0.0.1:${port}:5000/tcp ${model_name}
    else
        echo "Running ${model_name} on port ${port} ..."
        sudo docker run --rm --env-file ./docker.env -d -p 127.0.0.1:${port}:5000/tcp ${model_name}
    fi
}

function parse_arguments {
    while getopts ":hc:ad" opt; do
        case "${opt}" in
            h)
                echo "Usage: $0 [-h] [-c <container>] [-a]"
                echo "  -h  Show this help message"
                echo "  -c  Run only the specified container"
                echo "  -a  Run all containers"
                echo "  -d  Enable debug mode"
                exit 0
                ;;
            c)
                if [ -f "${OPTARG}/Dockerfile" ]; then
                    run_container "${OPTARG}/Dockerfile"
                    exit 0
                fi
                build_container "${OPTARG}"
                exit 0
                ;;
            a)
                echo "Running all containers ..."
                find . -name "Dockerfile" | while read dockerfile; do
                    run_container "${dockerfile}"
                done
                ;;
            d)
                echo "Debug mode enabled"
                debug_mode=true
                ;;
            *)
                echo "Invalid option: -${OPTARG}"
                exit 1
                ;;
        esac
    done
}

parse_arguments "$@"
