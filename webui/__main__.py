import os

import requests
from flask import Flask, render_template, request

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


@app.route("/api/chat", methods=["POST"])
def chat() -> dict:
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
