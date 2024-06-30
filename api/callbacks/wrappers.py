from io import BytesIO
from pathlib import Path
from typing import Union

from flask import Response, send_file
from werkzeug.wrappers.response import Response as BaseResponse

from .version import __version__


class ApiResponse(Response):
    """
    A custom API response class that includes the API version in the headers.
    """

    @staticmethod
    def from_dict(data: dict, status: int = 200) -> "ApiResponse":
        """
        Create a response from a dictionary.
        """
        return ApiResponse(
            response=data,
            status=status,
            mimetype="application/json",
            headers={
                "X-API-Version": __version__,
            },
        )

    @staticmethod
    def from_base_response(response: BaseResponse) -> "ApiResponse":
        """
        Create a response from a base response.
        """
        return ApiResponse(
            response=response.response,
            status=response.status_code,
            mimetype=response.mimetype,
            headers={
                **(response.headers or {}),
                "X-API-Version": __version__,
            },
        )

    @staticmethod
    def from_file(file_path: Union[str, Path]) -> "ApiResponse":
        """
        Create a response from a file path.
        """
        file_path = Path(file_path)
        try:
            with file_path.open("rb") as fp:
                return ApiResponse.from_base_response(
                    send_file(fp, as_attachment=True, mimetype="", download_name=file_path.name)
                )
        except FileNotFoundError:
            return ApiResponse.from_dict({"error": "file not found"}, status=404)
        except PermissionError:
            return ApiResponse.from_dict({"error": "permission denied"}, status=403)
        except Exception as exc:
            return ApiResponse.from_dict({"error": str(exc)}, status=500)

    @staticmethod
    def from_raw_bytes(data: bytes, mimetype: str, download_name: str) -> "ApiResponse":
        """
        Create a response from a data stream.
        """
        return ApiResponse.from_base_response(
            send_file(BytesIO(data), as_attachment=True, mimetype=mimetype, download_name=download_name)
        )
