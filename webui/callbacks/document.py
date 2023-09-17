from flask import render_template, request, session

from ..tools.db import open_sqlitedb_connection


def document_page() -> str:
    if not session.get("username", ""):
        return render_template("login.html", username="", error="You must be logged in to access this page")
    conn = open_sqlitedb_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS search_spaces (name TEXT UNIQUE)")
    cursor.execute("SELECT * FROM search_spaces")
    search_spaces = [row[0] for row in cursor.fetchall()]
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
