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
            "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        ),
        (
            "hentai.jpg",
            None,
            "f06c5dabb5d2476015e88fefafdcd986777a492d248acb3b67de2379ac90dc73",
        ),
        (
            "horses.jpg",
            {"censor": "yes"},
            "95a93c27639cefce637fed47583b4b6876c54c5e1b388bfb7a1af0e50a9271f5",
        ),
        (
            "hentai.jpg",
            {"censor": "yes"},
            "d5cbfd6e5997a7f6feaa7bdc82f753d5ef2dc4ce6d6fda08ee6d415f9df8b457",
        ),
    ],
    "agenet": [
        (
            "birthday.jpg",
            None,
            "04076a4b1659b0944a7c2ffb09140e7ec344f864cdd88fb617d05f007ef50d33",
        )
    ],
    "gfm34b2tt": [
        (
            "lamma.jpg",
            None,
            "0d899a14eeb6295652ff3cae981da1027d8203e164b490e57506cbff137d5853",
        ),
        (
            "lamma.jpg",
            {"background": "#ffcc00"},
            "af2e94873023a4f46a2073dcab9a32545ad3b6b711a867bf7d96857cd38b565e",
        ),
        (
            "lamma.jpg",
            {"background": "https://picsum.photos/id/1019/400"},
            "dbc9bd58469443df60ffabebbec2a8e415affbfd97a7ed8b0a186ea36ebec7bb",
        ),
        (
            "whisk.png",
            None,
            "b703544d3f6810a5e8298d0651ffd6cec8b358f6857fef8eb72a9fbc4e11b70b",
        ),
        (
            "whisk.png",
            {"background": "#ffcc00"},
            "83d6ed0585a09fe8aaf50fdde78daba2fa3d3acaa8a5cf5104ce0842f06ae5a8",
        ),
        (
            "whisk.png",
            {"background": "https://picsum.photos/id/1019/400"},
            "d08fce262f543892b09d11c6076dc555a81ccad213d0bb23f8437ea69bec66ec",
        ),
    ],
    "isnet": [
        (
            "valeria-lungu.jpg",
            None,
            "9d83cad44cadefeab7beba646ffe34ef9a9f00b58fd725d6c8eb9406de5691f0",
        ),
        (
            "valeria-lungu.jpg",
            {"background": "#ffcc00"},
            "a9aec95c0dbc5a93b67d3652a2d2dcaaea1231b626cfc7aa8e1e597d3a3a267b",
        ),
        (
            "valeria-lungu.jpg",
            {"background": "https://picsum.photos/id/1019/400"},
            "0f99b5a4ff23bca088f4feee0ea1dd93d91fbd62fb29a673ef8c13a3c42c4c6b",
        ),
        (
            "valeria-lungu.png",
            None,
            "76a58a0942f1052db6c18b3690fac98c996b07544c83853d32612a4c426194d8",
        ),
        (
            "valeria-lungu.png",
            {"background": "#ffcc00"},
            "f11585d2cced035eb3d22d697d76f30483398eefbd5b73bbcc1f2457da93c06e",
        ),
        (
            "valeria-lungu.png",
            {"background": "https://picsum.photos/id/1019/400"},
            "877bea22f1ca7a826719857657a403be5817016bec51f4b8cbf721486a4539ee",
        ),
    ],
    "srfbnet": [
        (
            "flowers.jpg",
            None,
            "0675e6b8eebdc90c8799a0ae544a0af5d84a6b85f9251ce1d7403a680172210b",
        ),
        (
            "flowers.jpg",
            {"scale": 2},
            "0675e6b8eebdc90c8799a0ae544a0af5d84a6b85f9251ce1d7403a680172210b",
        ),
        (
            "flowers.jpg",
            {"scale": 3},
            "dcace4dd63c802c4a6a54980de08ccb8696af5c52f788bf70338845395570aae",
        ),
        (
            "flowers.jpg",
            {"scale": 4},
            "71ec5b09ba3cd4b80854b452fe620d25bc9573932dfdddaf87961740d71902a7",
        ),
    ],
    "vitgpt2": [
        (
            "friends.jpg",
            None,
            "ec7c16e4da3d160e3a41c89675353f62b323085b83b853c87919b9790e2e0071",
        ),
        (
            "friends.jpg",
            {"max_length": 32},
            "ec7c16e4da3d160e3a41c89675353f62b323085b83b853c87919b9790e2e0071",
        ),
        (
            "friends.jpg",
            {"num_beams": 2},
            "1fee4c617b3c937457d4e1084b8777bc116a948843ab69ffddc14e3ce3dbc04a",
        ),
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
            info(f"JSON response: {json_response}")
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
                h = h.hexdigest()
                info(f"Image hash: {h}")
                if h != output:
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
                    h = h.hexdigest()
                    info(f"JSON hash: {h}")
                    if h != output:
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
