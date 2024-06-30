"""
Document processing callbacks using the file queue.
"""

import json
import logging
import os
import time
from hashlib import md5, sha256

from flask import request

from .. import __version__
from ..mimetypes import get_extension
from ..wrappers import ApiResponse
from .helpers import get_metadata_path, get_staged_path

logger = logging.getLogger(__name__)


def put_document() -> ApiResponse:
    """
    Upload a document to the file queue.
    The document will be hashed using API_FILE_HASHER (default: SHA256) and stored at /tmp/ai/staged/<hash>.<extension>.
    Also creates a metadata file at /tmp/ai/staged/<hash>.json with the following metadata:
    - type (str): the MIME type of the document
    - upload_time (time.time): the time the document was uploaded
    - processed (bool): whether the document has been processed yet
    Request details:
    - enctype: multipart/form-data
    - method: PUT
    - form data:
        - document: the document to upload
        - other: any other metadata to store with the document.
    """
    document_file = request.files.get("document")
    if not document_file:
        return ApiResponse.from_dict({"error": "missing document"}, status=400)

    document_type = document_file.mimetype
    document_data = document_file.read()

    document_hash = ({"MD5": md5, "SHA256": sha256}.get(os.getenv("API_FILE_HASHER", "SHA256").upper()) or sha256)()
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

    return ApiResponse.from_dict({"token": document_token})


def delete_document(document_token: str) -> ApiResponse:
    """
    Does not actually delete the document, but marks it for deletion by creating an empty application/x-delete file
    with the same token as the document under /tmp/ai/staged/<token>.delete path.
    Request details:
    - method: DELETE
    - path: /<endpoint>/<document_token>
    - form data (optional):
        - other: any other metadata to store with the document.
    - json data (optional):
        - other: any other metadata to store with the document.
    """
    if not document_token:
        return ApiResponse.from_dict({"error": "missing token"}, status=400)

    if request.mimetype == "application/json":
        if not isinstance(request.json, dict):
            return ApiResponse.from_dict({"error": "invalid json data"}, status=400)
        other_data = request.json
    elif request.mimetype == "application/x-www-form-urlencoded":
        other_data = request.form
    else:
        other_data = {}

    document_extension = ".delete"
    document_metadata = {
        **other_data,
        **{
            "type": "application/x-delete",
            "upload_time": time.time(),
            "processed": "false",
        },
    }
    meta_file = get_metadata_path(document_token)
    with meta_file.open("w") as fp:
        json.dump(document_metadata, fp)

    staged_file = get_staged_path(document_token, document_extension)
    with staged_file.open("w") as fp:
        fp.write("")

    return ApiResponse.from_dict({"token": document_token})
