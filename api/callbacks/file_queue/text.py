"""
Text processing callbacks using the file queue.
"""
import json
import os
import time
from hashlib import md5, sha256
from io import BytesIO
from pathlib import Path

from flask import Response, request, send_file

from .. import __version__
from ..mimetypes import get_mimetype
from .helpers import clean_files, get_metadata_path, get_prepared_paths, get_staged_path


def put_text() -> Response:
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
    if request.mimetype == "application/json":
        text_data = request.json.get("text", "")
        other_data = {k: v for k, v in request.json.items() if k != "text"}
    elif request.mimetype == "multipart/form-data":
        text_data = request.form.get("text", "")
        other_data = {k: v for k, v in request.form.items() if k != "text"}
    elif request.mimetype == "application/x-www-form-urlencoded":
        text_data = request.files.get("text")
        if text_data:
            text_data = text_data.read()
        other_data = request.form
    else:
        text_data = None
        other_data = {}

    if not text_data:
        return Response(
            json.dumps({"error": "missing text"}),
            status=400,
            mimetype="application/json",
        )

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

    return Response(
        json.dumps({"token": text_token, "version": __version__}),
        status=200,
        mimetype="application/json",
    )


def get_text(text_file: str) -> Response:
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
        return Response(
            json.dumps({"token": text_token, "error": "text not found"}),
            status=404,
            mimetype="application/json",
        )

    found_file = None
    for prepared_file in prepared_files:
        if prepared_file.name == text_file.name:
            found_file = prepared_file
            break

    if found_file is None:
        clean_files(text_token)
        return Response(
            json.dumps({"token": text_token, "error": "text not found"}),
            status=404,
            mimetype="application/json",
        )

    with found_file.open("rb") as fp:
        text_data = fp.read()

    found_file.unlink()
    if len(prepared_files) == 1:
        clean_files(text_token)

    if not text_data:
        return Response(
            json.dumps({"token": text_token, "error": "text output is empty"}),
            status=404,
            mimetype="application/json",
        )

    text_type = None
    try:
        text_type = get_mimetype(prepared_file.suffix)
    except ValueError:
        return Response(
            json.dumps({"token": text_token, "error": "unknown text type"}),
            status=404,
            mimetype="application/json",
        )

    return send_file(
        BytesIO(text_data),
        mimetype=text_type,
        as_attachment=True,
        download_name=text_token.name,
    )
