"""
CSV processing callbacks using the file queue.
"""

import json
import os
import time
from hashlib import md5, sha256
from pathlib import Path
from typing import Union

from flask import request

from .. import __version__
from ..mimetypes import get_mimetype
from ..wrappers import ApiResponse
from .helpers import clean_files, get_metadata_path, get_prepared_paths, get_staged_path


def put_csv() -> ApiResponse:
    """
    Upload a CSV file to the file queue.
    The CSV file will be hashed using API_FILE_HASHER (default: SHA256) and stored at /tmp/ai/staged/<hash>.<extension>.
    Also creates a metadata file at /tmp/ai/metadata/<hash>.json with the following metadata:
    - type (str): the MIME type of the CSV file
    - upload_time (time.time): the time the CSV file was uploaded
    - processed (bool): whether the CSV file has been processed yet
    Request details:
    - enctype: multipart/form-data
    - method: PUT
    - form data:
        - csv: the CSV file to upload
        - other: any other metadata to store with the CSV file.
    """
    csv_file = request.files.get("csv")
    if not csv_file:
        return ApiResponse.from_dict({"error": "missing csv"}, status=400)

    csv_type = csv_file.mimetype
    csv_data = csv_file.read()

    csv_hash = ({"MD5": md5, "SHA256": sha256}.get(os.getenv("API_FILE_HASHER", "SHA256").upper()) or sha256)()
    csv_hash.update(csv_data)
    csv_token = csv_hash.hexdigest()

    with open("/opt/app/mimetypes.json", "r") as fp:
        csv_extension = json.load(fp).get(csv_type, ".csv")

    csv_metadata = {
        **request.form,
        **{
            "type": csv_type,
            "upload_time": time.time(),
            "processed": "false",
        },
    }

    meta_file = get_metadata_path(csv_token)
    with meta_file.open("w") as fp:
        json.dump(csv_metadata, fp)

    staged_file = get_staged_path(csv_token, csv_extension)
    with staged_file.open("wb") as fp:
        fp.write(csv_data)

    return ApiResponse.from_dict({"token": csv_token})


def get_csv(csv_file: Union[str, Path]) -> ApiResponse:
    """
    Download a CSV file from the file queue.
    The CSV file will be retrieved from /tmp/ai/prepared/<hash>.<extension> and deleted.
    Sometimes, the models will produce multiple CSV files, so for csv_file you can use the <token>_<index>.<extension>
    format.
    Request details:
    - method: GET
    - path: /<endpoint>/<token>[_<index>].<extension>
    """
    csv_file = Path(csv_file)
    csv_token = csv_file.stem.split("_", 1)[0]

    prepared_files = get_prepared_paths(csv_token)

    if not prepared_files:
        clean_files(csv_token)
        return ApiResponse({"token": csv_token, "error": "csv not found"}, status=404)

    found_file = None
    for prepared_file in prepared_files:
        if prepared_file.name == csv_file.name:
            found_file = prepared_file
            break

    if found_file is None:
        clean_files(csv_token)
        return ApiResponse.from_dict({"token": csv_token, "error": "csv not found"}, status=404)

    with found_file.open("rb") as fp:
        csv_data = fp.read()

    found_file.unlink()
    if len(prepared_files) == 1:
        clean_files(csv_token)

    if not csv_data:
        return ApiResponse.from_dict({"token": csv_token, "error": "csv output is empty"}, status=404)

    csv_type = None
    try:
        csv_type = get_mimetype(csv_file)
    except ValueError:
        return ApiResponse.from_dict({"token": csv_token, "error": "unknown csv type"}, status=404)

    return ApiResponse.from_raw_bytes(csv_data, csv_type, csv_file.name)
