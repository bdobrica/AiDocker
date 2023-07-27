#!/usr/bin/env python3
import json
import os
import time
from hashlib import md5, sha256
from io import BytesIO
from pathlib import Path
from typing import List

from flask import Flask, Response, request, send_file

__version__ = "0.8.13"

try:
    with open("/opt/app/mimetypes.json", "r") as fp:
        MIMETYPES = json.load(fp)
except:
    MIMETYPES = {}

app = Flask(__name__)


def get_metadata_path(file_token: str) -> Path:
    STAGED_PATH = Path(os.getenv("STAGED_PATH", "/tmp/ai/staged"))
    return STAGED_PATH / (file_token + ".json")


def get_staged_path(file_token: str, file_suffix: str) -> Path:
    STAGED_PATH = Path(os.getenv("STAGED_PATH", "/tmp/ai/staged"))
    return STAGED_PATH / (file_token + "." + file_suffix.strip(".").lower())


def get_json_path(file_token: str, file_suffix: str = "json") -> Path:
    PREPARED_PATH = Path(os.getenv("PREPARED_PATH", "/tmp/ai/prepared"))
    return PREPARED_PATH / (file_token + "." + file_suffix.strip(".").lower())


def get_staged_paths(file_token: str) -> List[Path]:
    STAGED_PATH = Path(os.getenv("STAGED_PATH", "/tmp/ai/staged"))
    return [path for path in STAGED_PATH.glob(file_token + "*") if path.is_file() and path.suffix != ".json"]


def get_source_paths(file_token: str) -> List[Path]:
    SOURCE_PATH = Path(os.getenv("SOURCE_PATH", "/tmp/ai/source"))
    return [path for path in SOURCE_PATH.glob(file_token + "*") if path.is_file() and path.suffix != ".json"]


def get_prepared_paths(file_token: str) -> List[Path]:
    PREPARED_PATH = Path(os.getenv("PREPARED_PATH", "/tmp/ai/prepared"))
    return [path for path in PREPARED_PATH.glob(file_token + "*")]


def clean_files(file_token: str) -> None:
    paths = get_staged_paths(file_token) + get_source_paths(file_token) + get_prepared_paths(file_token)
    for path in paths:
        if path.exists():
            path.unlink()
    path = get_metadata_path(file_token)
    if path.exists():
        path.unlink()


def get_url(file: Path) -> str:
    for mimetype, suffix in MIMETYPES.items():
        if file.suffix == suffix:
            general_type, _ = mimetype.split("/")
            return f"/get/{general_type}/{file.name}"
    raise ValueError(f"Unknown file type: {file}")


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

    image_hash = ({"MD5": md5, "SHA256": sha256}.get(os.getenv("API_IMAGE_HASHER", "SHA256").upper()) or sha256)()
    image_hash.update(image_data)
    image_token = image_hash.hexdigest()

    with open("/opt/app/mimetypes.json", "r") as fp:
        image_extension = json.load(fp).get(image_type, ".jpg")

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


@app.route("/put/text", methods=["POST"])
def put_text():
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


@app.route("/get/image/<image_file>", methods=["GET"])
def get_image(image_file):
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
    for mimetype, suffix in MIMETYPES.items():
        if prepared_file.suffix == suffix:
            image_type = mimetype
            break
    if image_type is None:
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


@app.route("/get/json", methods=["POST"])
def get_json():
    lifetime = float(os.getenv("API_CLEANER_FILE_LIFETIME", "1800.0"))

    file_token = request.json.get("token")

    meta_file = get_metadata_path(file_token)
    if not meta_file.is_file():
        clean_files(file_token)
        return Response(
            json.dumps({"error": "missing file metadata", "version": __version__}),
            status=400,
            mimetype="application/json",
        )

    try:
        file_metadata = json.load(meta_file.open("r"))
    except:
        clean_files(file_token)
        return Response(
            json.dumps({"error": "corrupted image metadata", "version": __version__}),
            status=400,
            mimetype="application/json",
        )

    if float(file_metadata.get("upload_time", 0)) + lifetime < time.time():
        clean_files(file_token)
        return Response(
            json.dumps({"error": "token expired", "version": __version__}),
            status=400,
            mimetype="application/json",
        )

    json_file = get_json_path(file_token)
    if json_file.is_file():
        with json_file.open("r") as fp:
            try:
                json_data = json.load(fp)
            except:
                json_data = {}

        if not json_data:
            clean_files(file_token)
            return Response(
                json.dumps(
                    {"error": "invalid model output", "version": __version__},
                    status=400,
                    mimetype="application/json",
                )
            )

        json_data.update(
            {
                "token": file_token,
                "status": "success",
                "version": __version__,
                "inference_time": float(file_metadata.get("update_time", 0))
                - float(file_metadata.get("upload_time", 0)),
            }
        )
        clean_files(file_token)

        return Response(json.dumps(json_data), status=200, mimetype="application/json")

    prepared_files = get_prepared_paths(file_token)
    if prepared_files:
        output = {
            "token": file_token,
            "status": "success",
            "version": __version__,
            "inference_time": float(file_metadata.get("update_time", 0)) - float(file_metadata.get("upload_time", 0)),
        }
        if len(prepared_files) == 1:
            output["url"] = get_url(prepared_files[0])
        else:
            output["urls"] = [get_url(file) for file in prepared_files]

        return Response(json.dumps(output), status=200, mimetype="application/json")

    source_files = get_source_paths(file_token)
    if source_files:
        return Response(
            json.dumps({"wait": "true", "status": "processing", "version": __version__}),
            status=200,
            mimetype="application/json",
        )

    staged_files = get_staged_paths(file_token)
    if staged_files:
        return Response(
            json.dumps({"wait": "true", "status": "not queued", "version": __version__}),
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
