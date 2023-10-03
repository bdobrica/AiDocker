import os
import uuid

from flask import Flask, g, render_template, session

from .callbacks import (
    chat_api,
    chat_page,
    document_api,
    document_page,
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

chat_api = app.route("/api/chat", methods=["POST"])(chat_api)
chat_page = app.route("/chat/<search_space>")(chat_page)
document_api = app.route("/api/document", methods=["POST"])(document_api)
document_page = app.route("/document")(document_page)
documents_page = app.route("/documents/<search_space>/<page>", defaults={"page": 0})(documents_page)
login_page = app.route("/login", methods=["GET", "POST"])(login_page)
logout_page = app.route("/logout")(logout_page)
status_api = app.route("/api/status/<token>", methods=["GET"])(status_api)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=os.getenv("DEBUG", "false").lower() in ("true", "1", "on"))
