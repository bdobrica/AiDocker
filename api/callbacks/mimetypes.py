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
from functools import lru_cache, partial
from pathlib import Path
from typing import Dict, List, Union

import yaml
from yaml.loader import SafeLoader


def read_mimetypes(mimetype_file: Union[str, Path]) -> list:
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
    minetype_file = Path(mimetype_file) if not isinstance(mimetype_file, Path) else mimetype_file
    reader = {
        ".json": json.load,
        ".yaml": partial(yaml.load, Loader=SafeLoader),
    }
    try:
        with minetype_file.open("r") as fp:
            result = reader[minetype_file.suffix.lower()](fp)
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


@lru_cache
def load_mimetypes() -> List[Dict[str, str]]:
    mimetypes = []
    prefix = Path(__file__).parent.parent / "data"
    for file in ["mimetypes.yaml", "mimetypes.json"]:
        mimetypes = read_mimetypes(prefix / file)
        if mimetypes:
            return mimetypes
    raise FileNotFoundError(f"No mimetypes file found in {prefix}.")


def get_extension(mimetype: str) -> str:
    """
    Return the working (internal) extension for a given mimetype.
    :param mimetype: the mimetype to get the extension for, e.g. `text/csv`
    :return: the extension, e.g. `.csv`
    """
    mimetypes = load_mimetypes()
    for mimetype_ in mimetypes:
        if mimetype == mimetype_["type"]:
            return mimetype_["ext"]
    raise ValueError(f"Unknown file type: {mimetype}")


def get_mimetype(file_name: Union[str, Path]) -> str:
    """
    Get the mimetype for a given file using the extension.
    :param file: the file to get the mimetype for, e.g. `test.csv`
    :return: the mimetype, e.g. `text/csv`
    """
    mimetypes = load_mimetypes()
    file_name = Path(file_name) if not isinstance(file_name, Path) else file_name
    for mimetype in mimetypes:
        if file_name.suffix == mimetype["ext"]:
            return mimetype["type"]
    raise ValueError(f"Unknown file type for suffix: {file_name.suffix}")


def get_url(file_name: Union[str, Path]) -> str:
    """
    Given a file, return the URL to retrieve it.
    :param file: the file to get the URL for
    :return: the URL
    """
    mimetypes = load_mimetypes()
    file_name = Path(file_name) if not isinstance(file_name, Path) else file_name
    for mimetype in mimetypes:
        if file_name.suffix == mimetype["ext"]:
            return f"/get/{mimetype['file']}/{file_name.name}"
    raise ValueError(f"Unknown file type for suffix: {file_name.suffix}")
