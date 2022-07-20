import json
import mimetypes
import time
from pathlib import Path

import requests

TESTS = {
    "modnet": [
        ("valeria-lungu.jpg", None, ""),
    ],
    "u2net": [],
    "yolov4": [],
}


def put_image(url: str, image_path: Path, parameters: dict = None) -> dict:
    print(f"upload image {str(image_path)} to {url}")

    image_type, _ = mimetypes.guess_type(str(image_path))

    response = requests.request(
        "POST",
        url,
        headers={},
        data=parameters or {},
        files=[("image", (image_path.name, image_path.open("rb"), image_type))],
    )

    return response.json()


def get_json(url: str, image_token: str) -> dict:
    print(f"get response from {url} with token {image_token}")

    start_time = time.time()
    wait = True

    while wait:
        response = requests.request(
            "POST",
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"token": image_token}),
        )
        response_obj = response.json()
        wait = time.time() - start_time < 10.0
        if response_obj.get("wait") == "true":
            continue

    return response_obj


def get_image(url: str) -> bytes:
    print(f"download image from {url}")

    response = requests.request("GET", url)
    if response.status_code != 200:
        return None

    return response.content


def detect_model_urls(run_file: Path) -> dict:
    with run_file.open("r") as fp:
        urls = {}
        for line in fp:
            line = line.strip()
            if not line.startswith("docker run"):
                continue
            items = line.split()
            model = items[-1]
            url = "http://" + ":".join(items[-2].split(":")[:-1])
            urls[model] = url

    return urls


if __name__ == "__main__":
    urls = detect_model_urls(Path("../run.sh"))
    for model, tests in TESTS.items():
        url = urls.get(model)
        if url is None:
            # no tests for this model
            continue
        for image, parameters, output in tests:
            image_file = Path("data") / image

            # upload image
            response = put_image(f"{url}/put/image", image_file, parameters)
            image_token = response.get("token")

            json_response = get_json(f"{url}/get/json", image_token)
            if json_response.get("status") != "success":
                print(f"test fail for {model}!")

            if json_response.get("url"):
                image_url = json_response.get("url")
                image_content = get_image(f"{url}/{image_url}")
