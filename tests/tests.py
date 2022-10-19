import hashlib
import json
import mimetypes
import time
from pathlib import Path

import requests

PASSED: int = 0
FAILED: int = 0


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


TESTS = {
    "modnet": [
        (
            "valeria-lungu.jpg",
            None,
            "cc02dee408b5ef07b347d1386e6956390a941fb78c2047ccc02b7da147caa28f",
        ),
        (
            "valeria-lungu.jpg",
            {"background": "#ffcc00"},
            "df0a69a4342d54f5c2f11771aab05bd9bea3189d39cf66fcbcdeadc5563b6359",
        ),
        (
            "valeria-lungu.jpg",
            {"background": "https://picsum.photos/id/1019/400"},
            "c0de805cde11abba2382c8aa9b78a533b8b1a3e4673ca9f940d748c5cfcb67d1",
        ),
        (
            "valeria-lungu.png",
            None,
            "a4ccd904aa31bf90d8d2b465657a6b9f2e0003a6049ba4cedb5d944cd4d02fc2",
        ),
        (
            "valeria-lungu.png",
            {"background": "#ffcc00"},
            "6662ddc7e4da0d1ce36194ab29714e44d2604554ae349de15f2168b53e9972d1",
        ),
        (
            "valeria-lungu.png",
            {"background": "https://picsum.photos/id/1019/400"},
            "859ee627b4aaede612c04dd7775834e7877cc1e7f1070859af83ebce14f9005f",
        ),
    ],
    "u2net": [
        (
            "lamma.jpg",
            None,
            "f7633bb1f0026659c8302411b8db80352ec1b7a1a0af260bfb2bbdab776de628",
        ),
        (
            "lamma.jpg",
            {"background": "#ffcc00"},
            "6bb2d60714c8f7d3a28b0d19940ece404cfdb0c727d8a60650c53bf90d2a6f7d",
        ),
        (
            "lamma.jpg",
            {"background": "https://picsum.photos/id/1019/400"},
            "fb97996895f7d3492914d8b8921d6ee81ac5d211ad189d8a75fe71b240497d5b",
        ),
        (
            "whisk.png",
            None,
            "1887cc07de872a5f1b9db60e7aa23945fd71017a775d1d12a9889b736ebbe416",
        ),
        (
            "whisk.png",
            {"background": "#ffcc00"},
            "b794e7f561e44d133a5b2c85147d76042884e6356296a3fb9fa6b88234207b55",
        ),
        (
            "whisk.png",
            {"background": "https://picsum.photos/id/1019/400"},
            "67f965384b0b6f107a73c41f0b62600cc5b27386c09ffde2eda7d4306507c2be",
        ),
    ],
    "yolov4": [
        (
            "horses.jpg",
            None,
            "ab6ebcb3ce280c2ce6c082c03f34308c148b310488737344da81eeb0fbf345cc",
        ),
        (
            "person.jpg",
            None,
            "0a5829c8339b0a3acd919404255c83dea8b77af3d254a4a7b39a7167671b9e7b",
        ),
    ],
    "nudenet": [
        (
            "horses.jpg",
            None,
            "",
        ),
        (
            "hentai.jpg",
            None,
            "",
        ),
        (
            "horses.jpg",
            {"censor": "yes"},
            "",
        ),
        {
            "hentai.jpg",
            {"censor": "yes"},
            "",
        },
    ],
}


def info(message: str):
    print(f"[{Colors.OKBLUE} * {Colors.ENDC}] {message}")


def warn(message: str):
    print(f"[{Colors.WARNING} WARNING {Colors.ENDC}] {message}")


def error(message: str):
    global FAILED
    print(f"[{Colors.FAIL} ERROR {Colors.ENDC}] {message}")
    FAILED += 1


def ok(message: str):
    global PASSED
    print(f"[{Colors.OKGREEN} OK {Colors.ENDC}] {message}")
    PASSED += 1


def put_image(url: str, image_path: Path, parameters: dict = None) -> dict:
    info(f"Uploading image {str(image_path)} to {url} ...")

    image_type, _ = mimetypes.guess_type(str(image_path))

    try:
        response = requests.request(
            "POST",
            url,
            headers={},
            data=parameters or {},
            files=[
                ("image", (image_path.name, image_path.open("rb"), image_type))
            ],
        )

        response_obj = response.json()
    except:
        response_obj = {}

    return response_obj


def get_json(url: str, image_token: str) -> dict:
    info(f"Retrieving response from {url} using token {image_token} ...")

    start_time = time.time()
    wait = True

    while wait:
        time.sleep(0.5)
        try:
            response = requests.request(
                "POST",
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps({"token": image_token}),
            )
            response_obj = response.json()
        except:
            wait = False
            response_obj = {}
        wait = (
            wait
            and (time.time() - start_time < 10.0)
            and (response_obj.get("wait") == "true")
        )

    return response_obj


def get_image(url: str) -> bytes:
    info(f"Downloading result image from {url} ...")

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
    COUNTER = 0

    for model, tests in TESTS.items():
        url = urls.get(model)
        if url is None:
            warn(f"Could not find an active URL for model {model}. Skipping.")
            continue
        for image, parameters, output in tests:
            COUNTER += 1

            image_file = Path("data") / image

            # upload image
            response = put_image(f"{url}/put/image", image_file, parameters)
            image_token = response.get("token")
            if not image_token:
                error(
                    "Could not correctly upload the image"
                    f" {str(image_file)} to {url}/put/image."
                )
                continue

            json_response = get_json(f"{url}/get/json", image_token)
            if json_response.get("status") != "success":
                error(
                    "Could not correctly retrieve the info"
                    f" using {image_token} for image {str(image_file)}."
                )
                continue

            if json_response.get("url"):
                image_url = json_response.get("url")
                info(
                    f"The model {model} produces an output image under"
                    f" {url}/{image_url}."
                )
                image_content = get_image(f"{url}/{image_url}")
                h = hashlib.sha256()
                h.update(image_content)
                if h.hexdigest() != output:
                    error(
                        f"Failed testing model {model} with {str(image_file)}"
                        f" using parameters {str(parameters)}."
                    )
                else:
                    ok(
                        f"Passed testing model {model} with {str(image_file)}"
                        f" using parameters {str(parameters)}."
                    )
            else:
                if "results" in json_response:
                    info(f"The model {model} produces a JSON response.")
                    response_content = json_response["results"]
                    h = hashlib.sha256()
                    h.update(str(response_content).encode("utf-8"))
                    if h.hexdigest() != output:
                        error(
                            f"Failed testing model {model} with"
                            f" {str(image_file)} using parameters"
                            f" {str(parameters)}."
                        )
                    else:
                        ok(
                            f"Passed testing model {model} with"
                            f" {str(image_file)} using parameters"
                            f" {str(parameters)}."
                        )
                else:
                    warn(
                        f"The model {model} doesn't produce any useful"
                        " response."
                    )
    if FAILED == 0:
        if PASSED < COUNTER:
            warn(f"Passed {PASSED} tests / skipped {COUNTER - PASSED}.")
        else:
            ok(f"Passed all {PASSED} tests.")
    else:
        error(
            f"Failed {FAILED} tests / passed {PASSED} tests /"
            f" skipped {COUNTER - PASSED - FAILED} tests."
        )
