"""
Text processing callbacks using the file queue.
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


def put_text() -> ApiResponse:
    """
    Create a text file in the file queue from a string passed in the request.
    The text file will be hashed using API_TEXT_HASHER (default: SHA256) and stored at /tmp/ai/staged/<hash>.txt.
    Also creates a metadata file at /tmp/ai/metadata/<hash>.json with the following metadata:
    - type (str): the MIME type of the text file
    - upload_time (time.time): the time the text file was uploaded
    - processed (bool): whether the text file has been processed yet
    Request details:
    - method: PUT
    - form data (optional):
        - text: the text to upload
        - other: any other metadata to store with the text.
    - json data (optional):
        - text: the text to upload
        - other: any other metadata to store with the text.
    """
    text_data = None
    other_data = {}
    if request.mimetype == "application/json":
        if not isinstance(request.json, dict):
            return ApiResponse.from_dict({"error": "invalid JSON data"}, status=400)
        text_data = request.json.get("text", "")
        other_data = {k: v for k, v in request.json.items() if k != "text"}
    elif request.mimetype == "multipart/form-data":
        text_data = request.form.get("text", "")
        other_data = {k: v for k, v in request.form.items() if k != "text"}
    elif request.mimetype == "application/x-www-form-urlencoded":
        raw_data = request.files.get("text")
        if raw_data:
            text_data = raw_data.read().decode("utf8")
        other_data = request.form

    if not text_data:
        return ApiResponse.from_dict({"error": "missing text"}, status=400)

    text_hash = ({"MD5": md5, "SHA256": sha256}.get(os.getenv("API_TEXT_HASHER", "SHA256").upper()) or sha256)()
    text_hash.update(text_data.encode("utf8"))
    text_token = text_hash.hexdigest()
    text_extension = ".txt"

    text_metadata = {
        **other_data,
        **{
            "type": "text/plain",
            "upload_time": time.time(),
            "processed": "false",
        },
    }

    meta_file = get_metadata_path(text_token)
    with meta_file.open("w") as fp:
        json.dump(text_metadata, fp)

    staged_file = get_staged_path(text_token, text_extension)
    with staged_file.open("w") as fp:
        fp.write(text_data)

    return ApiResponse({"token": text_token, "version": __version__}, status=200)


def get_text(text_file: Union[str, Path]) -> ApiResponse:
    """
    Return a text file from the file queue.
    If the text file has been processed, the text is returned. The function is intended to be called with the URL/URLs
    returned by the endpoint handled with metadata.get_json function.
    Request details:
    - method: GET
    - path: /<endpoint>/<text_token>
    :param text_file: the text file to retrieve
    """
    text_file = Path(text_file)
    text_token = text_file.stem.split("_", 2)[0]

    prepared_files = get_prepared_paths(text_token)

    if not prepared_files:
        return ApiResponse.from_dict({"token": text_token, "error": "text not found"}, status=404)

    found_file = None
    for prepared_file in prepared_files:
        if prepared_file.name == text_file.name:
            found_file = prepared_file
            break

    if found_file is None:
        clean_files(text_token)
        return ApiResponse.from_dict({"token": text_token, "error": "text not found"}, status=404)

    with found_file.open("rb") as fp:
        text_data = fp.read()

    found_file.unlink()
    if len(prepared_files) == 1:
        clean_files(text_token)

    if not text_data:
        return ApiResponse.from_dict({"token": text_token, "error": "text output is empty"}, status=404)

    text_type = None
    try:
        text_type = get_mimetype(found_file)
    except ValueError:
        return ApiResponse.from_dict({"token": text_token, "error": "unknown text type"}, status=404)

    return ApiResponse.from_raw_bytes(text_data, text_type, text_file.name)
