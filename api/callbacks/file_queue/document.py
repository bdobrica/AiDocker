import json
import logging
import os
import time
from hashlib import md5, sha256

from flask import Response, request

from .. import __version__
from ..mimetypes import get_extension
from .helpers import get_metadata_path, get_staged_path

logger = logging.getLogger(__name__)


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

    document_extension = ".docx"
    try:
        document_extension = get_extension(document_type)
    except ValueError:
        logger.warning("unknown document type: %s. using default extension .docx", document_type)

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
