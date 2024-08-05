import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import IO, Iterator, Tuple

import pytest
from flask import Flask

from api import __version__
from api.callbacks.file_queue.image import get_image, put_image
from api.callbacks.mimetypes import get_mimetype

logger = logging.getLogger(__name__)


@pytest.fixture()
def app() -> Iterator[Flask]:
    app = Flask(__name__)
    app.config.update(
        {
            "TESTING": True,
        }
    )
    app.post("/put/image")(put_image)
    app.get("/get/image/<image_file>")(get_image)
    yield app


@pytest.fixture(
    params=[
        Path(__file__).parent.parent / "data" / "example-image.jpg",
        Path(__file__).parent.parent / "data" / "example-image.png",
    ]
)
def image_fp(request: pytest.FixtureRequest) -> Iterator[IO[bytes]]:
    image_path: Path = request.param
    with image_path.open("rb") as fp:
        yield fp


@pytest.fixture()
def file_queue() -> Iterator[Tuple[str, str, str]]:
    staged_path = os.getenv("STAGED_PATH")
    prepared_path = os.getenv("PREPARED_PATH")
    source_path = os.getenv("SOURCE_PATH")

    with tempfile.TemporaryDirectory() as staged_path, tempfile.TemporaryDirectory() as prepared_path, tempfile.TemporaryDirectory() as source_path:
        os.environ["STAGED_PATH"] = staged_path
        os.environ["SOURCE_PATH"] = source_path
        os.environ["PREPARED_PATH"] = prepared_path
        yield staged_path, source_path, prepared_path


def test_put_image(file_queue: Tuple[str, str, str], app: Flask, image_fp: IO[bytes]) -> None:
    staged_path, _, _ = file_queue
    image_content = image_fp.read()
    image_fp.seek(0)

    with app.test_client() as client:
        upload_time = time.time()

        logger.info(f"Uploading image {image_fp.name} ...")
        response = client.post(
            "/put/image",
            content_type="multipart/form-data",
            data={"image": (image_fp, image_fp.name)},
        )

        logger.info(f"Checking response ...")
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"
        assert response.headers["X-API-Version"] == __version__
        assert response.json is not None
        assert response.json.get("token") is not None
        token = response.json["token"]
        logger.info(f"File token: {token}")

        image_name = Path(image_fp.name)
        suffix = image_name.suffix.lower()
        image_mimetype = get_mimetype(image_name)

        logging.info("Checking if image was uploaded successfully ...")
        assert Path(f"{staged_path}/{token}{suffix}").exists()
        with Path(f"{staged_path}/{token}{suffix}").open("rb") as fp:
            assert fp.read() == image_content

        logging.info("Checking metadata ...")
        assert Path(f"{staged_path}/{token}.json").exists()
        with Path(f"{staged_path}/{token}.json").open("r") as fp:
            metadata = json.load(fp)
            assert metadata.get("type") == image_mimetype
            assert metadata.get("processed") == "false"
            assert abs(metadata.get("upload_time", 0) - upload_time) < 1.0


def test_get_image(file_queue: Tuple[str, str, str], app: Flask, image_fp: IO[bytes]) -> None:
    staged_path, _, prepared_path = file_queue
    image_content = image_fp.read()
    image_fp.seek(0)

    with app.test_client() as client:
        upload_time = time.time()

        logger.info(f"Uploading image {image_fp.name} ...")
        response = client.post(
            "/put/image",
            content_type="multipart/form-data",
            data={"image": (image_fp, image_fp.name)},
        )

        logger.info(f"Checking response ...")
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"
        assert response.headers["X-API-Version"] == __version__
        assert response.json is not None
        assert response.json.get("token") is not None
        token = response.json["token"]
        suffix = Path(image_fp.name).suffix.lower()
        logger.info(f"Image token: {token}")

        image_mime = get_mimetype(image_fp.name)

        staged_image_file = Path(f"{staged_path}/{token}{suffix}")
        prepared_image_file = Path(f"{prepared_path}/{token}{suffix}")

        logger.info(f"Checking if image was uploaded successfully ...")
        assert staged_image_file.exists()
        with staged_image_file.open("rb") as fp:
            assert fp.read() == image_content

        logger.info(f"Moving image to prepared path {prepared_image_file} ...")
        staged_image_file.rename(prepared_image_file)

        logger.info(f"Getting image {token} ...")
        response = client.get(f"/get/image/{token}{suffix}")

        logger.info(f"Checking response ...")
        assert response.status_code == 200
        assert response.headers["Content-Type"] == image_mime
        assert response.headers["X-API-Version"] == __version__
        assert response.headers["Content-Length"] == str(len(image_content))
        assert response.data == image_content

        logger.info(f"Checking if image was deleted ...")
        assert not staged_image_file.exists()
        assert not prepared_image_file.exists()
        assert not Path(f"{staged_path}/{token}.json").exists()
