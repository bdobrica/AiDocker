import hashlib
import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict

import requests
import yaml

PASSED: int = 0
FAILED: int = 0

with open("test_cases.yaml", "r") as fp:
    TESTS = yaml.load(fp, Loader=yaml.Loader)


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


def put_request(url: str, parameters: Dict[str, Any]) -> dict:
    files = []
    data = {}
    for key, value in parameters.items():
        possible_file = Path("data") / str(value)
        if possible_file.is_file():
            file_type, _ = mimetypes.guess_type(str(possible_file))
            files.append(
                (
                    key,
                    (possible_file.name, possible_file.open("rb"), file_type),
                )
            )
        else:
            data[key] = str(value)

    try:
        response = requests.request(
            "POST",
            url,
            headers={},
            data=data,
            files=files,
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
        time.sleep(1.0)
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
            and (time.time() - start_time < 60.0)
            and (response_obj.get("wait") == "true")
        )

    return response_obj


def get_request(url: str) -> bytes:
    info(f"Downloading result from {url} ...")

    response = requests.request("GET", url)
    if response.status_code != 200:
        return None

    return response.content


def detect_model_url(model: str) -> str:
    port_file = Path(f"../{model}/port.txt")
    if port_file.exists():
        with port_file.open("r") as fp:
            port = int(fp.read().strip())
    else:
        port = 5000

    return f"http://localhost:{port}"


if __name__ == "__main__":
    COUNTER = 0

    for model, tests in TESTS.items():
        url = detect_model_url(model)
        endpoint = tests.get("endpoint")
        if url is None:
            warn(f"Could not find an active URL for model {model}. Skipping.")
            continue
        for test in tests.get("test"):
            COUNTER += 1
            out_path = Path("out") / model / str(COUNTER)
            out_path.mkdir(parents=True, exist_ok=True)

            expected = test.get("expected")
            del test["expected"]

            # send the AI request
            response = put_request(f"{url}/{endpoint}", test)
            file_token = response.get("token")
            if not file_token:
                error(
                    "Could not correctly call put request with test"
                    f" {str(test)} to {url}/{endpoint}."
                )
                continue

            json_response = get_json(f"{url}/get/json", file_token)
            info(f"JSON response: {json_response}")
            with open(out_path / "response.json", "w") as fp:
                json.dump(json_response, fp, indent=4)

            if json_response.get("status") != "success":
                error(
                    "Could not correctly retrieve the info"
                    f" using {file_token} for test {str(test)}."
                )
                continue

            if json_response.get("url"):
                image_url = json_response.get("url")
                info(
                    f"The model {model} produces an output under"
                    f" {url}/{image_url}."
                )
                image_content = get_request(f"{url}/{image_url}")

                with open(out_path / Path(image_url).name, "wb") as fp:
                    fp.write(image_content)

                h = hashlib.sha256()
                h.update(image_content)
                h = h.hexdigest()
                info(f"Image hash: {h}")
                if h != expected:
                    error(f"Failed testing model {model} with {str(test)}.")
                else:
                    ok(f"Passed testing model {model} with {str(test)}.")
            else:
                if "results" in json_response:
                    info(f"The model {model} produces a JSON response.")
                    response_content = json_response["results"]
                    h = hashlib.sha256()
                    h.update(str(response_content).encode("utf-8"))
                    h = h.hexdigest()
                    info(f"JSON hash: {h}")
                    if h != expected:
                        error(
                            f"Failed testing model {model} with test {str(test)}."
                        )
                    else:
                        ok(
                            f"Passed testing model {model} with test {str(test)}."
                        )
                else:
                    warn(
                        f"The model {model} doesn't produce any useful response."
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
