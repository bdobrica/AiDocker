"""
Mimetype processing functions.

It produces a module-object called MIMETYPES, which is a list of dictionaries with the following keys:
- type (str): the mimetype, e.g. `text/csv`
- ext (str): the working (internal) extension, e.g. `.csv`
- file (str): the file to use for processing, e.g. `csv`
YAML definition files are searched for first, then JSON definition files. The first file found is used.
(see `read_mimetypes` for more details)
"""
import json
from functools import partial
from pathlib import Path

import yaml
from yaml.loader import SafeLoader

APP_PATH = Path("/opt/app")
MIMETYPES = []


def read_mimetypes(file: Path) -> list:
    """
    Mimetypes are specified in a YAML or JSON file by providing a standard extension and a file reference that allows selection of the correct callback. Here's an example:
    ```yaml
    - type: application/json
      ext: .json
      file: json
    - type: text/csv
      ext: .csv
      file: csv
    ```
    In this example, `application/json` files will be processed by the `<action>_json` callback, with the working (internal) extension `.json`, and `text/csv` files will be processed by the `<action>_csv` callback, with the working (internal) extension `.csv`.
    The JSON file does similar but without specifying the callback:
    ```json
    {
        "text/plain": ".txt",
        "text/csv": ".csv"
    }
    ```
    YAML is preferred over JSON because it allows for additional information to be added.
    :param file: the file to read
    :return: the list of mimetypes
    """
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
    """
    Return the working (internal) extension for a given mimetype.
    :param mimetype: the mimetype to get the extension for, e.g. `text/csv`
    :return: the extension, e.g. `.csv`
    """
    for mimetype_ in MIMETYPES:
        if mimetype == mimetype_["type"]:
            return mimetype_["ext"]
    raise ValueError(f"Unknown file type: {mimetype}")


def get_mimetype(file: Path) -> str:
    """
    Get the mimetype for a given file using the extension.
    :param file: the file to get the mimetype for, e.g. `test.csv`
    :return: the mimetype, e.g. `text/csv`
    """
    for mimetype in MIMETYPES:
        if file.suffix == mimetype["ext"]:
            return mimetype["type"]
    raise ValueError(f"Unknown file type: {file}")


def get_url(file: Path) -> str:
    """
    Given a file, return the URL to retrieve it.
    :param file: the file to get the URL for
    :return: the URL
    """
    for mimetype in MIMETYPES:
        if file.suffix == mimetype["ext"]:
            return f"/get/{mimetype['file']}/{file.name}"
    raise ValueError(f"Unknown file type: {file}")


for file in ["mimetypes.yaml", "mimetypes.json"]:
    MIMETYPES = read_mimetypes(APP_PATH / file)
    if MIMETYPES:
        break
