from typing import Optional

from flask import render_template, session

from ..orm import Document, SearchSpace


def documents_page(search_space: Optional[str] = None, page: int = 0) -> str:
    if not session.get("username", ""):
        return render_template("login.html", username="", error="You must be logged in to access this page")
    search_spaces = SearchSpace.select()
    if not search_space:
        raise ValueError("Missing search_space field")
    page = int(page)

    documents, page, pages = Document.select_paginated(page=page)

    return render_template(
        "documents.html",
        search_spaces=search_spaces,
        documents=4 * documents,
        pages=pages,
        current_page_no=page,
    )
