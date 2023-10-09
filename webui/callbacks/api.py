import os

import requests
from flask import request, session
from werkzeug.utils import secure_filename

from ..orm import Document, SearchSpace


def add_document_api() -> dict:
    if not session.get("username", ""):
        raise ValueError("You must be logged in to access this page")

    if "document" not in request.files:
        raise ValueError("Missing document field")

    document = request.files["document"]
    if document.filename == "":
        raise ValueError("Missing document name")

    search_space = request.form.get("search_space", "")
    if search_space == "":
        raise ValueError("Missing search_space field")

    search_space = next(SearchSpace.select(id=search_space), None)
    if not search_space:
        search_space = SearchSpace(
            id=search_space,
            name=search_space,
        ).insert()

    index_model_host = os.getenv("INDEX_MODEL_HOST", "localhost:5000")

    docname = secure_filename(document.filename)

    response = requests.put(
        f"http://{index_model_host}/put/document",
        files={"document": (docname, document.stream, document.mimetype)},
        data={"search_space": search_space},
    )

    response.raise_for_status()
    response = response.json()
    response = {"token": "test"}

    _ = Document(
        search_space=search_space,
        token=response["token"],
        name=docname,
    ).insert()

    return {"token": response["token"]}


def delete_document_api(document_id: int) -> dict:
    if not session.get("username", ""):
        raise ValueError("You must be logged in to access this page")

    document = next(Document.select(id=document_id), None)
    if not document:
        raise ValueError("Invalid document id")

    index_model_host = os.getenv("INDEX_MODEL_HOST", "localhost:5000")

    response = requests.delete(
        f"http://{index_model_host}/delete/document",
        data={"token": document.token, "search_space": document.search_space},
    )

    response.raise_for_status()
    response = response.json()
    response = {"token": "test"}

    document.delete()

    return {"token": response["token"]}


def status_api(token: str) -> dict:
    if not session.get("username", ""):
        raise ValueError("You must be logged in to access this page")

    index_model_host = os.getenv("INDEX_MODEL_HOST", "localhost:5000")
    response = requests.get(f"http://{index_model_host}/get/status/", json={"token": token})
    response.raise_for_status()
    response = response.json()
    return {"status": response["status"]}


def chat_api() -> dict:
    data = request.get_json()
    if "prompt" not in data:
        raise ValueError("Missing prompt field")

    chat_model_host = os.getenv("CHAT_MODEL_HOST", "localhost:5000")

    response = requests.put(
        f"http://{chat_model_host}/put/text",
        json={"text": data["prompt"], "search_space": "moby"},
    )
    response.raise_for_status()
    response = response.json()

    return {"answer": response["results"][0]["answer"]}
