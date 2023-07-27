import json
import os
import time
from hashlib import md5, sha256

from flask import Response, request

from .. import __version__
from .helpers import get_metadata_path, get_prepared_paths, get_staged_path


def put_document() -> Response:
    document_file = request.files.get("document")
    if not document_file:
        return Response(
            json.dumps({"error": "missing document"}),
            status=400,
            mimetype="application/json",
        )

    document_type = document_file.mimetype
    document_data = document_file.read()

    document_hash = ({"MD5": md5, "SHA256": sha256}.get(os.getenv("API_CSV_HASHER", "SHA256").upper()) or sha256)()
    document_hash.update(document_data)
    document_token = document_hash.hexdigest()

    with open("/opt/app/mimetypes.json", "r") as fp:
        document_extension = json.load(fp).get(document_type, ".docx")

    document_metadata = {
        **request.form,
        **{
            "type": document_type,
            "upload_time": time.time(),
            "processed": "false",
        },
    }

    meta_file = get_metadata_path(document_token)
    with meta_file.open("w") as fp:
        json.dump(document_metadata, fp)

    staged_file = get_staged_path(document_token, document_extension)
    with staged_file.open("wb") as fp:
        fp.write(document_data)

    return Response(
        json.dumps({"token": document_token, "version": __version__}),
        status=200,
        mimetype="application/json",
    )
