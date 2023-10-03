import os

import requests
from flask import render_template, request, session
from werkzeug.utils import secure_filename

from ..orm import Document, SearchSpace


def document_page() -> str:
    if not session.get("username", ""):
        return render_template("login.html", username="", error="You must be logged in to access this page")
    search_spaces = SearchSpace.select()
    return render_template("document.html", search_spaces=search_spaces)


def document_api() -> dict:
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

    search_spaces = SearchSpace.select(id=search_space)
    if search_spaces:
        search_space = search_spaces[0]
    else:
        search_space = SearchSpace(
            id=search_space,
            name=search_space,
        ).insert()

    index_model_host = os.getenv("INDEX_MODEL_HOST", "localhost:5000")

    docname = secure_filename(document.filename)

    # response = requests.put(
    #    f"http://{index_model_host}/put/document",
    #    files={"document": (docname, document.stream, document.mimetype)},
    #    data={"search_space": search_space},
    # )

    # response.raise_for_status()
    # response = response.json()
    response = {"token": "test"}

    _ = Document(
        search_space=search_space,
        token=response["token"],
        name=docname,
    ).insert()

    return {"token": response["token"]}
