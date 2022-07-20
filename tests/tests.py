import mimetypes
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
    image_type, _ = mimetypes.guess_type(str(image_path))

    response = requests.request(
        "POST",
        url,
        headers={},
        data=parameters or {},
        files=[("image", (image_path.name, image_path.open("rb"), image_type))],
    )

    return response


def get_image(url: str, image_path: Path) -> bool:
    response = requests.request("GET", url)
    if response.status_code != 200:
        return False

    with image_path.open("wb") as fp:
        fp.write(response.content)


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
    for model, tests in TESTS:
        url = urls.get(model)
        if url is None:
            # no tests for this model
            continue
        for image, parameters, output in tests:
            image_file = Path("data") / image

            # upload image
            response = put_image(f"{url}/put/image", image_file, parameters)

            print(response)
