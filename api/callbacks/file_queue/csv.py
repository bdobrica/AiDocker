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


def put_csv() -> Response:
    csv_file = request.files.get("csv")
    if not csv_file:
        return Response(
            json.dumps({"error": "missing csv"}),
            status=400,
            mimetype="application/json",
        )

    csv_type = csv_file.mimetype
    csv_data = csv_file.read()

    csv_hash = ({"MD5": md5, "SHA256": sha256}.get(os.environ.get("API_CSV_HASHER", "SHA256").upper()) or sha256)()
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

    return Response(
        json.dumps({"token": csv_token, "version": __version__}),
        status=200,
        mimetype="application/json",
    )


def get_csv(csv_file) -> Response:
    csv_file = Path(csv_file)
    csv_token = csv_file.stem.split("_", 2)[0]

    prepared_files = get_prepared_paths(csv_token)

    if not prepared_files:
        clean_files(csv_token)
        return Response(
            json.dumps({"token": csv_token, "error": "csv not found"}),
            status=404,
            mimetype="application/json",
        )

    found_file = None
    for prepared_file in prepared_files:
        if prepared_file.name == csv_file.name:
            found_file = prepared_file
            break

    if found_file is None:
        clean_files(csv_token)
        return Response(
            json.dumps({"token": csv_token, "error": "csv not found"}),
            status=404,
            mimetype="application/json",
        )

    with found_file.open("rb") as fp:
        csv_data = fp.read()

    found_file.unlink()
    if len(prepared_files) == 1:
        clean_files(csv_token)

    if not csv_data:
        return Response(
            json.dumps({"token": csv_token, "error": "csv output is empty"}),
            status=404,
            mimetype="application/json",
        )

    csv_type = None
    try:
        csv_type = get_mimetype(csv_file)
    except ValueError:
        return Response(
            json.dumps({"token": csv_token, "error": "unknown csv type"}),
            status=404,
            mimetype="application/json",
        )

    return send_file(
        BytesIO(csv_data),
        mimetype=csv_type,
        as_attachment=True,
        download_name=csv_file.name,
    )
