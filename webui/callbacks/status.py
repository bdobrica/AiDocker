import os

import requests
from flask import session


def status_api(token: str) -> dict:
    if not session.get("username", ""):
        raise ValueError("You must be logged in to access this page")

    index_model_host = os.getenv("INDEX_MODEL_HOST", "localhost:5000")
    response = requests.get(f"http://{index_model_host}/get/status/", json={"token": token})
    response.raise_for_status()
    response = response.json()
    return {"status": response["status"]}
