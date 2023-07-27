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
    text_data = request.form.get("text", "")

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
        **request.form,
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


def get_test(text_file) -> Response:
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
