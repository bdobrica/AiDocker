import json
import logging
import os
import random
import string
import tempfile
import time
from pathlib import Path
from typing import IO, Iterator, Tuple

import pytest
from flask import Flask

from api import __version__
from api.callbacks.file_queue.metadata import get_json

logger = logging.getLogger(__name__)


@pytest.fixture()
def app() -> Iterator[Flask]:
    app = Flask(__name__)
    app.config.update(
        {
            "TESTING": True,
        }
    )
    app.get("/get/json/<file_token>")(get_json)
    yield app


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


@pytest.fixture()
def file_token() -> Iterator[str]:
    yield "".join(random.choice(string.ascii_letters + string.digits) for _ in range(32))


def test_get_json(file_queue: Tuple[str, str, str], app: Flask, file_token: str) -> None:
    staged_path, _, prepared_path = file_queue

    with app.test_client() as client:
        upload_time = time.time()

        logger.info("testing missing file token")
        response = client.get(f"/get/json/{file_token}")
        assert response.status_code == 400
        assert response.json is not None
        assert response.json["error"] == "missing file metadata"

        logger.info("testing getting file metadata")
        meta_file = Path(staged_path) / f"{file_token}.json"
        with meta_file.open("w+") as fp:
            json.dump(
                {
                    "upload_time": upload_time,
                    "update_time": upload_time,
                    "version": __version__,
                    "processed": False,
                },
                fp,
            )

        response = client.get(f"/get/json/{file_token}")
        assert response.status_code == 200
        assert response.json is not None
        assert response.json["upload_time"] == upload_time
        assert response.json["update_time"] == upload_time
        assert response.json["version"] == __version__
        assert response.json["processed"] is False

        logger.info("testing model output is json file")
        prepared_file = Path(prepared_path) / f"{file_token}.json"
        inference_results = list(range(10))
        with prepared_file.open("w+") as fp:
            json.dump(
                {
                    "results": inference_results,
                },
                fp,
            )
        response = client.get(f"/get/json/{file_token}")
        assert response.status_code == 200
        assert response.json is not None
        assert response.json["version"] == __version__
        assert set(response.json["results"]) == set(inference_results)
        assert abs(float(response.json["inference_time"])) < 0.1

        logger.info("when output is json, metadata and output are cleaned")
        assert not prepared_file.exists()
        assert not meta_file.exists()

        logger.info("testing model output is image file")
        with meta_file.open("w+") as fp:
            json.dump(
                {
                    "upload_time": upload_time,
                    "update_time": upload_time,
                    "version": __version__,
                    "processed": False,
                },
                fp,
            )
        prepared_image = prepared_file.with_suffix(".png")
        prepared_image.touch()
        response = client.get(f"/get/json/{file_token}")
        assert response.status_code == 200
        assert response.json is not None
        assert response.json["version"] == __version__
        assert Path(response.json["url"]).name == prepared_image.name

        logger.info("when output is image, metadata is not cleaned waiting for next action")
        assert prepared_image.exists()
        assert meta_file.exists()

        prepared_image.unlink()
        meta_file.unlink()
