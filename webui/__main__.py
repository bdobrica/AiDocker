import os
import sqlite3
import uuid
from pathlib import Path
from typing import Union

import requests
from flask import Flask, Response, g, redirect, render_template, request, session
from werkzeug.utils import secure_filename

app = Flask(__name__)
Flask.secret_key = uuid.uuid4().hex


def open_sqlitedb_connection():
    db = getattr(g, "_database", None)
    if db is None:
        database_path = Path(os.getenv("DATABASE_PATH", "/opt/db/database.db"))
        database_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(database_path)
        setattr(g, "_database", db)
    return db


@app.teardown_appcontext
def close_sqlitedb_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


@app.route("/")
def index() -> str:
    username = session.get("username", "")
    return render_template("index.html", username=username)


@app.route("/login", methods=["GET", "POST"])
def login_page() -> Union[Response, str]:
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == os.getenv("WEB_UI_USERNAME", "admin") and password == os.getenv("WEB_UI_PASSWORD", "admin"):
            session["username"] = username
            return redirect(request.referrer)
        else:
            session["username"] = ""
            del session["username"]
            return render_template("login.html", username=username, error="Invalid username or password")
    return render_template("login.html", username="", error="")


@app.route("/logout")
def logout_page() -> Response:
    session["username"] = ""
    del session["username"]
    return redirect(request.referrer)


@app.route("/document")
def document_page() -> str:
    if not session.get("username", ""):
        return render_template("login.html", username="", error="You must be logged in to access this page")
    conn = open_sqlitedb_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS search_spaces (name TEXT UNIQUE)")
    cursor.execute("SELECT * FROM search_spaces")
    search_spaces = [row[0] for row in cursor.fetchall()]
    return render_template("document.html", search_spaces=search_spaces)


@app.route("/chat/<search_space>")
def chat_page(search_space: str) -> str:
    return render_template("chat.html", search_space=search_space)


@app.route("/api/document", methods=["POST"])
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
    if not session.get("username", ""):
        raise ValueError("You must be logged in to access this page")

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
    print("index", os.getenv("INDEX_MODEL_HOST", "localhost:5000"))
    print("chat", os.getenv("CHAT_MODEL_HOST", "localhost:5000"))
    app.run(host="0.0.0.0", debug="run")
