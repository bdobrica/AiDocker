import os

import requests
from flask import render_template, request


def chat_page(search_space: str) -> str:
    return render_template("chat.html", search_space=search_space)


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
