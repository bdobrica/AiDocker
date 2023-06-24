import json
import os
import time
from pathlib import Path

from flask import Response, request

from .helpers import (
    clean_files,
    get_json_path,
    get_metadata_path,
    get_prepared_paths,
    get_source_paths,
    get_staged_paths,
)
from .mimetypes import get_url

__version__ = "0.8.12"


def get_json():
    lifetime = float(os.environ.get("API_CLEANER_FILE_LIFETIME", "1800.0"))

    file_token = request.json.get("token")

    meta_file = get_metadata_path(file_token)
    if not meta_file.is_file():
        clean_files(file_token)
        return Response(
            json.dumps({"error": "missing file metadata", "version": __version__}),
            status=400,
            mimetype="application/json",
        )

    try:
        file_metadata = json.load(meta_file.open("r"))
    except:
        clean_files(file_token)
        return Response(
            json.dumps({"error": "corrupted image metadata", "version": __version__}),
            status=400,
            mimetype="application/json",
        )

    if float(file_metadata.get("upload_time", 0)) + lifetime < time.time():
        clean_files(file_token)
        return Response(
            json.dumps({"error": "token expired", "version": __version__}),
            status=400,
            mimetype="application/json",
        )

    json_file = get_json_path(file_token)
    if json_file.is_file():
        with json_file.open("r") as fp:
            try:
                json_data = json.load(fp)
            except:
                json_data = {}

        if not json_data:
            clean_files(file_token)
            return Response(
                json.dumps(
                    {"error": "invalid model output", "version": __version__},
                    status=400,
                    mimetype="application/json",
                )
            )

        json_data.update(
            {
                "token": file_token,
                "status": "success",
                "version": __version__,
                "inference_time": float(file_metadata.get("update_time", 0))
                - float(file_metadata.get("upload_time", 0)),
            }
        )
        clean_files(file_token)

        return Response(json.dumps(json_data), status=200, mimetype="application/json")

    prepared_files = get_prepared_paths(file_token)
    if prepared_files:
        output = {
            "token": file_token,
            "status": "success",
            "version": __version__,
            "inference_time": float(file_metadata.get("update_time", 0)) - float(file_metadata.get("upload_time", 0)),
        }
        if len(prepared_files) == 1:
            output["url"] = get_url(prepared_files[0])
        else:
            output["urls"] = [get_url(file) for file in prepared_files]

        return Response(json.dumps(output), status=200, mimetype="application/json")

    source_files = get_source_paths(file_token)
    if source_files:
        return Response(
            json.dumps({"wait": "true", "status": "processing", "version": __version__}),
            status=200,
            mimetype="application/json",
        )

    staged_files = get_staged_paths(file_token)
    if staged_files:
        return Response(
            json.dumps({"wait": "true", "status": "not queued", "version": __version__}),
            status=200,
            mimetype="application/json",
        )

    return Response(
        json.dumps({"error": "unknown token", "version": __version__}),
        status=400,
        mimetype="application/json",
    )
