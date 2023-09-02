import os

import requests
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat() -> dict:
    data = request.get_json()
    if "prompt" not in data:
        raise ValueError("Missing text field")

    chat_model_host = os.getenv("CHAT_MODEL_HOST", "localhost:5000")

    response = requests.put(
        f"http://{chat_model_host}/put/text",
        json={"text": data["prompt"]},
    )
    response.raise_for_status()
    response = response.json()

    return {"answer": response["answer"]}


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug="run")
