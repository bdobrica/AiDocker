#!/usr/bin/env python3
"""
Main entrypoint for the API server.

This file is responsible for loading the configuration, registering the endpoints, and starting the server.
It loads the configuration from /opt/app/container.yaml, which is a YAML file with the following structure:
- input (list): a list of input endpoints to register
    - endpoint (str): the endpoint to register
    - queue (str): the queue to use for this endpoint (file or zero)
- output (list): a list of output endpoints to register
    - endpoint (str): the endpoint to register
    - queue (str): the queue to use for this endpoint (file or zero)
"""
import inspect
import json
import logging
import os
from pathlib import Path

import yaml
from flask import Flask, Response, g

from daemon import ZeroQueueMixin

from .callbacks import __version__, file_queue, zero_queue

app = Flask(__name__)

if __name__ == "__main__":
    config_path = Path("/opt/app/container.yaml")
    with config_path.open("r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    endpoints = []
    input = config.get("input", [])
    if not isinstance(input, list):
        input = [input]
    endpoints.extend(input)
    output = config.get("output", [])
    if not isinstance(output, list):
        output = [output]
    endpoints.extend(output)

    do_debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "on")
    if do_debug:
        app.logger.setLevel(logging.DEBUG)

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
        signature = inspect.signature(callback)
        for param in signature.parameters:
            if param == "self":
                continue
            parts.append(f"<{param}>")
        endpoint_path = "/" + "/".join(part for part in parts if part)
        app.logger.info(
            "Registering endpoint %s / %s with queue %s",
            endpoint_path,
            method,
            endpoint["queue"],
        )
        app.route(endpoint_path, methods=[method])(callback)

    if load_zmq:
        with app.app_context():
            app.logger.info("Loading ZeroMQ")
            zmq_mixin = ZeroQueueMixin()
            app.config["zmq_worker_timeout"] = zmq_mixin.worker_timeout
            app.config["zmq_client_address"] = zmq_mixin.client_address

    @app.route("/", methods=["GET", "POST"])
    def get_root():
        return Response(
            json.dumps({"version": __version__}),
            status=200,
            mimetype="application/json",
        )

    app.run(host="0.0.0.0", debug=do_debug)
