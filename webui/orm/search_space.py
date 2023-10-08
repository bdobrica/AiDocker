import re
from typing import Optional

from pydantic import root_validator

from ..tools import OrmBase


class SearchSpace(OrmBase):
    id: Optional[str] = None
    name: str
    language: Optional[str] = None

    @root_validator(pre=True)
    def validate_name(cls, values):
        id = values.get("id")
        name = values.get("name")
        if not id and not name:
            raise ValueError("Missing id or name field")
        if id is None:
            slug_re = re.compile(r"[^a-z0-9]+")
            values["id"] = slug_re.sub("-", name.lower())
        return values
