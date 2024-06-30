"""
Image file queue callbacks.
"""

import json
import logging
import os
import time
from hashlib import md5, sha256
from pathlib import Path
from typing import Union

from flask import request

from .. import __version__
from ..mimetypes import get_extension, get_mimetype
from ..wrappers import ApiResponse
from .helpers import clean_files, get_metadata_path, get_prepared_paths, get_staged_path

logger = logging.getLogger(__name__)


def put_image() -> ApiResponse:
    """
    Upload an image to the file queue.
    The image will be hashed using API_FILE_HASHER (default: SHA256) and stored at /tmp/ai/staged/<hash>.<extension>.
    Also creates a metadata file at /tmp/ai/staged/<hash>.json with the following metadata:
    - type (str): the MIME type of the image
    - upload_time (time.time): the time the image was uploaded
    - processed (bool): whether the image has been processed yet
    Request details:
    - enctype: multipart/form-data
    - method: PUT
    - form data:
        - image: the image to upload
        - other: any other metadata to store with the image.
    """
    image_file = request.files.get("image")
    if not image_file:
        return ApiResponse.from_dict({"error": "missing image"}, status=400)

    image_type = image_file.mimetype
    image_data = image_file.read()

    image_hash = ({"MD5": md5, "SHA256": sha256}.get(os.getenv("API_FILE_HASHER", "SHA256").upper()) or sha256)()
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

    return ApiResponse.from_dict({"token": image_token})


def get_image(image_file: Union[str, Path]) -> ApiResponse:
    """
    Get an image from the file queue.
    If the image has been processed, the image is returned. The function is intended to be called with the URL/URLs
    returned by the endpoint handled with metadata.get_json function.
    Request details:
    - method: GET
    - path: /<endpoint>/<image_name>
    :param image_file: the image file to get
    :return: the image
    """
    image_file = Path(image_file)
    image_token = image_file.stem.split("_", 2)[0]

    prepared_files = get_prepared_paths(image_token)

    if not prepared_files:
        clean_files(image_token)
        return ApiResponse.from_dict({"token": image_token, "error": "image not found"}, status=404)

    found_file = None
    for prepared_file in prepared_files:
        if prepared_file.name == image_file.name:
            found_file = prepared_file
            break

    if found_file is None:
        clean_files(image_token)
        return ApiResponse.from_dict({"token": image_token, "error": "image not found"}, status=404)

    with found_file.open("rb") as fp:
        image_data = fp.read()

    found_file.unlink()
    if len(prepared_files) == 1:
        clean_files(image_token)

    if not image_data:
        return ApiResponse.from_dict({"token": image_token, "error": "image output is empty"}, status=404)

    image_type = None
    try:
        image_type = get_mimetype(image_file)
    except ValueError:
        return ApiResponse.from_dict({"token": image_token, "error": "unknown image type"}, status=404)

    return ApiResponse.from_raw_bytes(image_data, image_type, image_file.name)
