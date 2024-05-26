#!/bin/bash

APP_PATH="/opt/app"
CONFIG_PATH="${APP_PATH}/container.yaml"

export PYTHONPATH="${PYTHONPATH}:${APP_PATH}"

function yq { python3 -c "import json,sys,yaml;print(json.dumps(yaml.safe_load(sys.stdin)))" | jq "$@"; }

# Start queue daemons
cat "${CONFIG_PATH}" | yq -r '.input[].queue'| sort -u | while read queue; do
    case $queue in
        "file")
            echo "Starting file queue cleaner ..."
            /usr/bin/env python3 -m daemon -d cleaner & ;;
        "zero")
            echo "Starting ZeroMQ broker ..."
            /usr/bin/env python3 -m daemon -d broker & ;;
        *)
            echo "Unknown queue type: $queue" >&2; exit 1;;
    esac
done

cat "${CONFIG_PATH}" | yq -r '.daemons[]' | sort -u | while read daemon; do
    python_args=$(echo $daemon | sed -e 's/\([a-z_]\+\)\/\([a-z_]\+\)/\1 -d \2/g' -e 's/.\+/-m \0/')
    echo "Starting daemon: python3 $python_args"
    /usr/bin/env python3 $python_args &
done

# Trusty old infinite loop
while true; do sleep 1; done
