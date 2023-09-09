import os

import requests
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/file")
def file_page():
    return render_template("file.html")


@app.route("/chat")
def chat_page():
    return render_template("chat.html")


@app.route("/api/document", methods=["POST"])
def document_api() -> dict:
    if "document" not in request.files:
        raise ValueError("Missing document field")

    document = request.files["document"]
    if document.filename == "":
        raise ValueError("Missing document name")

    index_model_host = os.getenv("INDEX_MODEL_HOST", "localhost:5000")

    docname = secure_filename(document.filename)

    response = requests.put(
        f"http://{index_model_host}/put/document",
        files={"document": (docname, document.stream, document.mimetype)},
        data={"search_space": request.form.get("search_space", "")},
    )

    response.raise_for_status()
    response = response.json()
    return {"token": response["token"]}


@app.route("/api/status/<token>", methods=["GET"])
def status_api(token: str) -> dict:
    index_model_host = os.getenv("INDEX_MODEL_HOST", "localhost:5000")
    response = requests.get(f"http://{index_model_host}/get/status/", json={"token": token})
    response.raise_for_status()
    response = response.json()
    return {"status": response["status"]}


@app.route("/api/chat", methods=["POST"])
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

    print("answer", response["results"][0]["answer"])

    return {"answer": response["results"][0]["answer"]}


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug="run")
