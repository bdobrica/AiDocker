import json
import logging
import os
import time
from hashlib import md5, sha256
from io import BytesIO
from pathlib import Path

from flask import Response, request, send_file

from .. import __version__
from ..mimetypes import get_extension, get_mimetype
from .helpers import clean_files, get_metadata_path, get_prepared_paths, get_staged_path

logger = logging.getLogger(__name__)


def put_image() -> Response:
    image_file = request.files.get("image")
    if not image_file:
        return Response(
            json.dumps({"error": "missing image"}),
            status=400,
            mimetype="application/json",
        )

    image_type = image_file.mimetype
    image_data = image_file.read()

    image_hash = ({"MD5": md5, "SHA256": sha256}.get(os.environ.get("API_IMAGE_HASHER", "SHA256").upper()) or sha256)()
    image_hash.update(image_data)
    image_token = image_hash.hexdigest()

    image_extension = ".jpg"
    try:
        image_extension = get_extension(image_type)
    except ValueError:
        logger.warning("unknown image type: %s. using default extension .jpg", image_type)

    image_metadata = {
        **request.form,
        **{
            "type": image_type,
            "upload_time": time.time(),
            "processed": "false",
        },
    }

    meta_file = get_metadata_path(image_token)
    with meta_file.open("w") as fp:
        json.dump(image_metadata, fp)

    staged_file = get_staged_path(image_token, image_extension)
    with staged_file.open("wb") as fp:
        fp.write(image_data)

    return Response(
        json.dumps({"token": image_token, "version": __version__}),
        status=200,
        mimetype="application/json",
    )


def get_image(image_file) -> Response:
    image_file = Path(image_file)
    image_token = image_file.stem.split("_", 2)[0]

    prepared_files = get_prepared_paths(image_token)

    if not prepared_files:
        clean_files(image_token)
        return Response(
            json.dumps({"token": image_token, "error": "image not found"}),
            status=404,
            mimetype="application/json",
        )

    found_file = None
    for prepared_file in prepared_files:
        if prepared_file.name == image_file.name:
            found_file = prepared_file
            break

    if found_file is None:
        clean_files(image_token)
        return Response(
            json.dumps({"token": image_token, "error": "image not found"}),
            status=404,
            mimetype="application/json",
        )

    with found_file.open("rb") as fp:
        image_data = fp.read()

    found_file.unlink()
    if len(prepared_files) == 1:
        clean_files(image_token)

    if not image_data:
        return Response(
            json.dumps({"token": image_token, "error": "image output is empty"}),
            status=404,
            mimetype="application/json",
        )

    image_type = None
    try:
        image_type = get_mimetype(image_data)
    except ValueError:
        return Response(
            json.dumps({"token": image_token, "error": "unknown image type"}),
            status=404,
            mimetype="application/json",
        )

    return send_file(
        BytesIO(image_data),
        mimetype=image_type,
        as_attachment=True,
        download_name=image_file.name,
    )
