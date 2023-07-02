import json
from functools import partial
from pathlib import Path

import yaml
from yaml.loader import SafeLoader

APP_PATH = Path("/opt/app")
MIMETYPES = []


def read_mimetypes(file: Path) -> list:
    reader = {
        ".json": json.load,
        ".yaml": partial(yaml.load, Loader=SafeLoader),
    }
    try:
        with file.open("r") as fp:
            result = reader[file.suffix.lower()](fp)
            if isinstance(result, dict):
                return [
                    {
                        "file": key.split("/", 2)[0],
                        "type": key,
                        "ext": value,
                    }
                    for key, value in result.items()
                ]
            elif isinstance(result, list):
                return result
    except:
        pass
    return []


def get_extension(mimetype: str) -> str:
    for mimetype_ in MIMETYPES:
        if mimetype == mimetype["mime"]:
            return mimetype["ext"]
    raise ValueError(f"Unknown file type: {file}")


def get_mimetype(file: Path) -> str:
    for mimetype in MIMETYPES:
        if file.suffix == mimetype["ext"]:
            return mimetype["type"]
    raise ValueError(f"Unknown file type: {file}")


def get_url(file: Path) -> str:
    for mimetype in MIMETYPES:
        if file.suffix == mimetype["ext"]:
            return f"/get/{mimetype['file']}/{file.name}"
    raise ValueError(f"Unknown file type: {file}")


for file in ["mimetypes.yaml", "mimetypes.json"]:
    MIMETYPES = read_mimetypes(APP_PATH / file)
    if MIMETYPES:
        break
