import os
import sqlite3

import requests
from flask import Flask, g, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)


def open_sqlitedb_connection():
    db = getattr(g, "_database", None)
    if db is None:
        database_path = os.getenv("DATABASE_PATH", "database.db")
        db = sqlite3.connect(database_path)
        setattr(g, "_database", db)
    return db


@app.teardown_appcontext
def close_sqlitedb_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/document")
def document_page():
    conn = open_sqlitedb_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS search_spaces (name TEXT UNIQUE)")
    cursor.execute("SELECT * FROM search_spaces")
    search_spaces = cursor.fetchall()
    return render_template("document.html", search_spaces=search_spaces)


@app.route("/chat/<search_space>")
def chat_page():
    return render_template("chat.html")


@app.route("/api/document", methods=["POST"])
def document_api() -> dict:
    if "document" not in request.files:
        raise ValueError("Missing document field")

    document = request.files["document"]
    if document.filename == "":
        raise ValueError("Missing document name")

    search_space = request.form.get("search_space", "")
    if search_space == "":
        raise ValueError("Missing search_space field")

    conn = open_sqlitedb_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS search_spaces (name TEXT UNIQUE)")
    cursor.execute("INSERT OR IGNORE INTO search_spaces VALUES (?)", (search_space,))
    conn.commit()

    index_model_host = os.getenv("INDEX_MODEL_HOST", "localhost:5000")

    docname = secure_filename(document.filename)

    response = requests.put(
        f"http://{index_model_host}/put/document",
        files={"document": (docname, document.stream, document.mimetype)},
        data={"search_space": search_space},
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
