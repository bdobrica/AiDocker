import os
import time
from hashlib import md5, sha256

from .helpers import ApiHandler, ApiResponse, CSVRequest


def put_csv(request: CSVRequest) -> ApiResponse:
    csv_file = request.csv
    if not csv_file:
        return ApiResponse(error="missing csv", status=400)

    csv_type = csv_file.content_type
    csv_data = csv_file.file.read()

    csv_hash = ({"MD5": md5, "SHA256": sha256}.get(os.getenv("API_FILE_HASHER", "SHA256").upper()) or sha256)()
    csv_hash.update(csv_data)
    csv_token = csv_hash.hexdigest()

    with open("/opt/app/mimetypes.json", "r") as fp:
        csv_extension = json.load(fp).get(csv_type, ".csv")

    csv_metadata = {
        **request.form,
        **{
            "type": csv_type,
            "upload_time": time.time(),
            "processed": "false",
        },
    }

    meta_file = get_metadata_path(csv_token)
    with meta_file.open("w") as fp:
        json.dump(csv_metadata, fp)

    staged_file = get_staged_path(csv_token, csv_extension)
    with staged_file.open("wb") as fp:
        fp.write(csv_data)


def get_csv() -> ApiResponse:
    return ApiResponse()
