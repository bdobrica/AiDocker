from typing import Optional

from flask import render_template, session

from ..orm import Document, SearchSpace


def documents_page(search_space: Optional[str] = None, page: int = 0) -> str:
    if not session.get("username", ""):
        return render_template("login.html", username="", error="You must be logged in to access this page")

    if search_space is not None:
        search_space = SearchSpace(name=search_space)
    page = int(page)

    documents, page, pages = Document.select_paginated(page=page)

    return render_template(
        "documents.html",
        page_title="Document Management",
        search_space=search_space,
        documents=4 * documents,
        pages=pages,
        current_page_no=page,
    )


def add_document_page() -> str:
    if not session.get("username", ""):
        return render_template("login.html", username="", error="You must be logged in to access this page")
    search_spaces = SearchSpace.select()
    return render_template("add_document.html", page_title="Add Document", search_spaces=search_spaces)


def delete_document_page(document_id: int) -> str:
    if not session.get("username", ""):
        return render_template("login.html", username="", error="You must be logged in to access this page")
    document = next(Document.select(id=document_id), None)
    if not document:
        return render_template("error.html", error="Document not found")
    return render_template("delete_document.html", page_title="Delete Document", document=document)
