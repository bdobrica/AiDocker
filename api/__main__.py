#!/usr/bin/env python3
import json
import os
from pathlib import Path

import yaml
import zmq
from flask import Flask, Response, g

from daemon import ZeroQueueMixin

from .callbacks import __version__, file_queue, zero_queue

app = Flask(__name__)

if __name__ == "__main__":
    config_path = Path("/opt/app/container.yaml")
    with config_path.open("r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    endpoints = []
    if isinstance(config["input"], list):
        endpoints.extend(config["input"])
    else:
        endpoints.append(config["input"])
    if isinstance(config["output"], list):
        endpoints.extend(config["output"])
    else:
        endpoints.append(config["output"])

    load_zmq = False
    for endpoint in endpoints:
        parts = endpoint["endpoint"].strip("/").split("/")
        method = parts[0].upper()
        callback_name = "_".join(parts[:2])
        module = {
            "file": file_queue,
            "zero": zero_queue,
        }.get(endpoint["queue"], None)
        if module is None or not hasattr(module, callback_name):
            continue
        if endpoint["queue"] == "zero":
            load_zmq = True
        callback = getattr(module, callback_name)
        app.logger.info("Registering endpoint %s / %s with queue %s", endpoint["endpoint"], method, endpoint["queue"])
        app.route(endpoint["endpoint"], methods=[method])(callback)

    if load_zmq:
        with app.app_context():
            app.logger.info("Loading ZeroMQ")
            zmq_mixin = ZeroQueueMixin()
            context = zmq.Context()
            g.zmq_context = context
            g.zmq_worker_timeout = zmq_mixin.worker_timeout
            g.zmq_client_address = zmq_mixin.client_address

    @app.route("/", methods=["GET", "POST"])
    def get_root():
        return Response(
            json.dumps({"version": __version__}),
            status=200,
            mimetype="application/json",
        )

    app.run(host="0.0.0.0", debug=(os.getenv("DEBUG", "false").lower() in ("true", "1", "on")))
