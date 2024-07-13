from typing import Annotated, Optional

from fastapi import FastAPI, Form, UploadFile
from pydantic import BaseModel


class ApiResponse(BaseModel):
    error: str
    status: int


class ApiHandler(FastAPI):
    pass


class CSVRequest(BaseModel):
    csv: Optional[UploadFile] = None
