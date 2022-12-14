#!/usr/bin/env python3
import base64
import json
import os
import time
from hashlib import md5, sha256
from io import BytesIO
from pathlib import Path

from flask import Flask, Response, request, send_file

app = Flask(__name__)


__version__ = "0.8.7"


def file_paths(image_token, image_extension=None):
    STAGED_PATH = Path(os.environ.get("STAGED_PATH", "/tmp/ai/staged"))
    SOURCE_PATH = Path(os.environ.get("SOURCE_PATH", "/tmp/ai/source"))
    PREPARED_PATH = Path(os.environ.get("PREPARED_PATH", "/tmp/ai/prepared"))

    meta_file = STAGED_PATH / (image_token + ".json")
    if image_extension is None and meta_file.is_file():
        with meta_file.open() as fp:
            try:
                image_meta = json.load(fp)
            except:
                image_meta = {}
        image_extension = image_meta.get("extension", None)

    json_file = PREPARED_PATH / (image_token + ".json")
    if image_extension is not None:
        staged_file = STAGED_PATH / (image_token + image_extension)
        source_file = SOURCE_PATH / (image_token + image_extension)
        prepared_file = PREPARED_PATH / (image_token + image_extension)
    else:
        staged_file = STAGED_PATH.glob(image_token + ".*")
        source_file = SOURCE_PATH.glob(image_token + ".*")
        prepared_file = PREPARED_PATH.glob(image_token + ".*")

    return {
        "meta_file": meta_file,
        "json_file": json_file,
        "staged_file": staged_file,
        "source_file": source_file,
        "prepared_file": prepared_file,
    }


def clean_files(image_token, image_extension=None):
    paths = file_paths(image_token, image_extension)
    for path in paths.values():
        if isinstance(path, Path):
            if path.is_file():
                path.unlink()
        else:
            for path_ in path:
                if path_.is_file():
                    path_.unlink()


@app.route("/", methods=["GET", "POST"])
def get_root():
    return Response(
        json.dumps({"version": __version__}),
        status=200,
        mimetype="application/json",
    )


@app.route("/put/image", methods=["POST"])
def put_image():
    image_file = request.files.get("image")
    if not image_file:
        return Response(
            json.dumps({"error": "missing image"}),
            status=400,
            mimetype="application/json",
        )

    image_type = image_file.mimetype
    image_data = image_file.read()

    image_hash = (
        {"MD5": md5, "SHA256": sha256}.get(
            os.environ.get("API_IMAGE_HASHER", "SHA256").upper()
        )
        or sha256
    )()
    image_hash.update(image_data)
    image_token = image_hash.hexdigest()

    with open("/opt/app/mimetypes.json", "r") as fp:
        image_extension = json.load(fp).get(image_type, ".jpg")

    image_metadata = {
        **request.form,
        **{
            "type": image_type,
            "extension": image_extension,
            "upload_time": time.time(),
            "processed": "false",
        },
    }

    paths = file_paths(image_token, image_extension)

    meta_file = paths["meta_file"]
    with meta_file.open("w") as fp:
        json.dump(image_metadata, fp)

    staged_file = paths["staged_file"]
    with staged_file.open("wb") as fp:
        fp.write(image_data)

    return Response(
        json.dumps({"token": image_token, "version": __version__}),
        status=200,
        mimetype="application/json",
    )


@app.route("/put/text", methods=["POST"])
def put_text():
    text_data = request.form.get("text", "")
    if not text_data:
        return Response(
            json.dumps({"error": "missing text"}),
            status=400,
            mimetype="application/json",
        )

    text_hash = (
        {"MD5": md5, "SHA256": sha256}.get(
            os.environ.get("API_TEXT_HASHER", "SHA256").upper()
        )
        or sha256
    )()
    text_hash.update(text_data.encode("utf8"))
    text_token = text_hash.hexdigest()
    text_extension = ".txt"

    text_metadata = {
        **request.form,
        **{
            "type": "text/plain",
            "extension": text_extension,
            "upload_time": time.time(),
            "processed": "false",
        },
    }

    paths = file_paths(text_token, text_extension)

    meta_file = paths["meta_file"]
    with meta_file.open("w") as fp:
        json.dump(text_metadata, fp)

    staged_file = paths["staged_file"]
    with staged_file.open("w") as fp:
        fp.write(text_data)

    return Response(
        json.dumps({"token": text_token, "version": __version__}),
        status=200,
        mimetype="application/json",
    )


@app.route("/get/image/<image_file>", methods=["GET"])
def get_image(image_file):
    image_file_path = Path(image_file)
    image_token = image_file_path.stem
    image_extension = image_file_path.suffix

    paths = file_paths(image_token, image_extension)
    prepared_file = paths["prepared_file"]

    if not prepared_file.is_file():
        return Response("image not found", status=404)

    with prepared_file.open("rb") as fp:
        image_data = fp.read()
    if not image_data:
        return Response("image output is empty", status=404)

    meta_file = paths["meta_file"]
    if meta_file.is_file():
        with meta_file.open() as fp:
            try:
                image_metadata = json.load(fp)
            except:
                image_metadata = {}

    image_type = image_metadata.get("type", "image/jpeg")

    clean_files(image_token, image_extension)

    return send_file(
        BytesIO(image_data),
        mimetype=image_type,
        as_attachment=True,
        download_name=image_token + image_extension,
    )


@app.route("/get/json", methods=["POST"])
def get_json():
    lifetime = float(os.environ.get("API_CLEANER_FILE_LIFETIME", "1800.0"))

    image_token = request.json.get("token")
    paths = file_paths(image_token)

    meta_file = paths["meta_file"]
    if not meta_file.is_file():
        clean_files(image_token)
        return Response(
            json.dumps(
                {"error": "missing file metadata", "version": __version__}
            ),
            status=400,
            mimetype="application/json",
        )

    try:
        image_metadata = json.load(meta_file.open("r"))
    except:
        clean_files(image_token)
        return Response(
            json.dumps(
                {"error": "corrupted image metadata", "version": __version__}
            ),
            status=400,
            mimetype="application/json",
        )

    if float(image_metadata.get("upload_time", 0)) + lifetime < time.time():
        clean_files(image_token)
        return Response(
            json.dumps({"error": "token expired", "version": __version__}),
            status=400,
            mimetype="application/json",
        )

    if image_metadata.get("extension") is None:
        clean_files(image_token)
        return Response(
            json.dumps(
                {"error": "invalid image extension", "version": __version__}
            ),
            status=400,
            mimetype="application/json",
        )

    json_file = paths["json_file"]
    prepared_file = paths["prepared_file"]

    if json_file.is_file():
        with json_file.open("r") as fp:
            try:
                json_data = json.load(fp)
            except:
                json_data = {}

        if not json_data:
            clean_files(image_token)
            return Response(
                json.dumps(
                    {"error": "invalid model output", "version": __version__},
                    status=400,
                    mimetype="application/json",
                )
            )

        json_data.update(
            {"token": image_token, "status": "success", "version": __version__}
        )
        json_file.unlink()

        return Response(
            json.dumps(json_data), status=200, mimetype="application/json"
        )

    staged_file = paths["staged_file"]
    source_file = paths["source_file"]

    if prepared_file.is_file():
        if staged_file.is_file():
            staged_file.unlink()

        return Response(
            json.dumps(
                {
                    "url": "/get/image/{image_token}.{image_extension}".format(
                        image_token=image_token,
                        image_extension=prepared_file.suffix[1:],
                    ),
                    "status": "success",
                    "version": __version__,
                }
            ),
            status=200,
            mimetype="application/json",
        )

    if source_file.is_file():
        return Response(
            json.dumps(
                {"wait": "true", "status": "processing", "version": __version__}
            ),
            status=200,
            mimetype="application/json",
        )

    if staged_file.is_file():
        return Response(
            json.dumps(
                {"wait": "true", "status": "not queued", "version": __version__}
            ),
            status=200,
            mimetype="application/json",
        )

    return Response(
        json.dumps({"error": "unknown token", "version": __version__}),
        status=400,
        mimetype="application/json",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
