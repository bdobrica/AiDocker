from flask import render_template, request


def chat_page(search_space: str) -> str:
    return render_template("chat.html", page_title="Chat", search_space=search_space)
