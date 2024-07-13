import json
import logging
import os
import tempfile
import time
from pathlib import Path

import pytest
from flask import Flask

from api import __version__
from api.callbacks.file_queue.csv import get_csv, put_csv

logger = logging.getLogger(__name__)

CSV_CONTENT = """a,b,c
1,2,3
4,5,6"""


@pytest.fixture()
def app():
    app = Flask(__name__)
    app.config.update(
        {
            "TESTING": True,
        }
    )
    app.post(("/put/csv"))(put_csv)
    app.get(("/get/csv/<csv_file>"))(get_csv)
    yield app


@pytest.fixture()
def csv_fp():
    with tempfile.NamedTemporaryFile(suffix=".csv") as fp:
        fp.write(CSV_CONTENT.encode("utf-8"))
        fp.seek(0)
        yield fp


@pytest.fixture()
def file_queue():
    staged_path = os.getenv("STAGED_PATH")
    prepared_path = os.getenv("PREPARED_PATH")
    source_path = os.getenv("SOURCE_PATH")

    with tempfile.TemporaryDirectory() as staged_path, tempfile.TemporaryDirectory() as prepared_path, tempfile.TemporaryDirectory() as source_path:
        os.environ["STAGED_PATH"] = staged_path
        os.environ["SOURCE_PATH"] = source_path
        os.environ["PREPARED_PATH"] = prepared_path
        yield staged_path, source_path, prepared_path


def test_put_csv(file_queue, app, csv_fp):
    staged_path, _, _ = file_queue
    with app.test_client() as client:
        upload_time = time.time()

        logging.info("Uploading CSV file ...")
        response = client.post("/put/csv", data={"csv": csv_fp})

        logging.info("Checking response ...")
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"
        assert response.headers["X-API-Version"] == __version__
        assert response.json.get("token") is not None
        token = response.json["token"]
        logging.info("File token: %s", token)

        logging.info("Checking if file was upload successfully ...")
        assert Path(f"{staged_path}/{token}.csv").exists()
        with Path(f"{staged_path}/{token}.csv").open("rb") as fp:
            assert fp.read().decode("utf-8") == CSV_CONTENT

        logging.info("Checking metadata ...")
        assert Path(f"{staged_path}/{token}.json").exists()
        with Path(f"{staged_path}/{token}.json").open("r") as fp:
            metadata = json.load(fp)
            assert metadata.get("type") == "text/csv"
            assert metadata.get("processed") == "false"
            assert abs(metadata.get("upload_time") - upload_time) < 1.0


def test_get_csv(file_queue, app, csv_fp):
    staged_path, _, prepared_path = file_queue
    with app.test_client() as client:
        logging.info("Uploading CSV file ...")
        response = client.post("/put/csv", data={"csv": csv_fp})

        logging.info("Checking response ...")
        assert response.status_code == 200
        assert response.json.get("token") is not None
        token = response.json["token"]
        logging.info("File token: %s", token)

        logging.info("Checking if file was upload successfully ...")
        assert Path(f"{staged_path}/{token}.csv").exists()
        logging.info("Checking metadata ...")
        assert Path(f"{staged_path}/{token}.json").exists()
        logging.info("Checking move to prepared (simulate processing) ...")
        Path(f"{staged_path}/{token}.csv").rename(f"{prepared_path}/{token}.csv")

        logging.info("Downloading CSV file ...")
        response = client.get(f"/get/csv/{token}.csv")

        logging.info("Checking response ...")
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
        assert response.headers["X-API-Version"] == __version__

        logging.info("Checking CSV content ...")
        assert response.data.decode("utf-8") == CSV_CONTENT

        logging.info("Checking if files were cleaned up ...")
        assert not Path(f"{staged_path}/{token}.csv").exists()
        assert not Path(f"{staged_path}/{token}.json").exists()
        assert not Path(f"{prepared_path}/{token}.csv").exists()
        assert not Path(f"{prepared_path}/{token}.json").exists()
