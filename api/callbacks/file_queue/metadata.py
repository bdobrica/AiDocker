"""
File queue calls are asynchronous, so you need to poll the API to see if your file is ready. This contains the callback
functions that you need to do that.
"""
import json
import os
import time

from flask import Response

from .. import __version__
from ..mimetypes import get_url
from .helpers import (
    clean_files,
    get_metadata_path,
    get_prepared_path,
    get_prepared_paths,
)


def get_json(file_token: str) -> Response:
    """
    Checks the status of a file in the file queue:
    - reads the file metadata to check if the file has expired or not; files expire after API_CLEANER_FILE_LIFETIME; if
        the file has expired, it is deleted and an error is returned
    - if the file has not expired, checks if the file has been processed and a JSON output file has been created; if
        the file has been processed, the JSON output file is returned
    - if the file has been processed, but there's no JSON output file, some output files may have been created; if there
        are such files, `url` or `urls` is added to the metadata and returned
    - if the file has not been processed, the metadata is returned
    Request details:
    - method: GET
    - path: /<endpoint>/<file_token>
    """
    lifetime = float(os.getenv("API_CLEANER_FILE_LIFETIME", "1800.0"))

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
            json.dumps({"error": "corrupted metadata", "version": __version__}),
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

    json_file = get_prepared_path(file_token)
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
                **file_metadata,
            }
        )
        clean_files(file_token)

        return Response(json.dumps(json_data), status=200, mimetype="application/json")

    file_metadata.update(
        {
            "token": file_token,
            "version": __version__,
            "inference_time": float(file_metadata.get("update_time", 0)) - float(file_metadata.get("upload_time", 0)),
        }
    )
    prepared_files = get_prepared_paths(file_token)
    if prepared_files:
        if len(prepared_files) == 1:
            file_metadata["url"] = get_url(prepared_files[0])
        else:
            file_metadata["urls"] = [get_url(file) for file in prepared_files]

    return Response(json.dumps(file_metadata), status=200, mimetype="application/json")
