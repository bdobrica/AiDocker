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
from api.callbacks.file_queue.document import delete_document, put_document
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
    app.post(("/put/document"))(put_document)
    app.get(("/delete/document/<document_file>"))(delete_document)
    yield app


@pytest.fixture(
    params=[
        Path(__file__).parent.parent / "data" / "example-document.docx",
        Path(__file__).parent.parent / "data" / "example-document.pdf",
    ]
)
def document_fp(request: pytest.FixtureRequest) -> Iterator[IO[bytes]]:
    document_path: Path = request.param
    with document_path.open("rb") as fp:
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


def test_put_document(file_queue: Tuple[str, str, str], app: Flask, document_fp: IO[bytes]) -> None:
    staged_path, _, _ = file_queue
    document_content = document_fp.read()
    document_fp.seek(0)

    with app.test_client() as client:
        upload_time = time.time()

        logging.info("Uploading CSV file ...")
        response = client.post("/put/document", data={"document": document_fp})

        logging.info("Checking response ...")
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"
        assert response.headers["X-API-Version"] == __version__
        assert response.json is not None
        assert response.json.get("token") is not None
        token = response.json["token"]
        logging.info("File token: %s", token)

        document_name = Path(document_fp.name)
        suffix = document_name.suffix.lower()
        document_mimetype = get_mimetype(document_name)

        logging.info("Checking if file was upload successfully ...")
        assert Path(f"{staged_path}/{token}{suffix}").exists()
        with Path(f"{staged_path}/{token}{suffix}").open("rb") as fp:
            assert fp.read() == document_content

        logging.info("Checking metadata ...")
        assert Path(f"{staged_path}/{token}.json").exists()
        with Path(f"{staged_path}/{token}.json").open("r") as fp:
            metadata = json.load(fp)
            assert metadata.get("type") == document_mimetype
            assert metadata.get("processed") == "false"
            assert abs(metadata.get("upload_time") - upload_time) < 1.0


def test_delete_document(file_queue: Tuple[str, str, str], app: Flask, document_fp: IO[bytes]) -> None:
    staged_path, _, _ = file_queue
    document_fp.seek(0)

    with app.test_client() as client:
        upload_time = time.time()

        logging.info("Uploading the document file %s ...", document_fp.name)
        response = client.post("/put/document", data={"document": document_fp})

        logging.info("Checking response ...")
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"
        assert response.headers["X-API-Version"] == __version__
        assert response.json is not None
        assert response.json.get("token") is not None
        token = response.json["token"]
        suffix = Path(document_fp.name).suffix.lower()
        logging.info("File token: %s", token)

        document_file = Path(token).with_suffix(suffix)

        logging.info("Deleting document %s ...", document_fp.name)
        response = client.get(f"/delete/document/{document_file}")

        logging.info("Checking response ...")
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"
        assert response.headers["X-API-Version"] == __version__
        assert response.json is not None
        assert response.json.get("token") == token
        assert response.json.get("error") is None

        logging.info("Checking if the .delete file was created ...")
        assert Path(f"{staged_path}/{token}.delete").exists()
        assert Path(f"{staged_path}/{token}.json").exists()
