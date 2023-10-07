import os
import uuid

from flask import Flask, g, render_template, session

from .callbacks import (
    add_document_api,
    add_document_page,
    chat_api,
    chat_page,
    delete_document_api,
    delete_document_page,
    documents_page,
    login_page,
    logout_page,
    status_api,
)
from .tools.db import dispose_db_engine

app = Flask(__name__)
Flask.secret_key = uuid.uuid4().hex


@app.route("/")
def index() -> str:
    username = session.get("username", "")
    return render_template("index.html", username=username)


dispose_db_engine = app.teardown_appcontext(dispose_db_engine)

_ = app.route("/api/chat", methods=["POST"])(chat_api)
_ = app.route("/api/document", methods=["DELETE"])(delete_document_api)
_ = app.route("/api/document", methods=["PUT"])(add_document_api)
_ = app.route("/api/status/<token>", methods=["GET"])(status_api)
_ = app.route("/chat/<search_space>")(chat_page)
_ = app.route("/documents")(documents_page)
_ = app.route("/documents/<search_space>")(documents_page)
_ = app.route("/documents/<search_space>/page=<page>")(documents_page)
_ = app.route("/documents/add")(add_document_page)
_ = app.route("/documents/delete/<document_id>")(delete_document_page)
_ = app.route("/documents/page=<page>")(documents_page)
_ = app.route("/login", methods=["GET", "POST"])(login_page)
_ = app.route("/logout")(logout_page)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=os.getenv("DEBUG", "false").lower() in ("true", "1", "on"))
