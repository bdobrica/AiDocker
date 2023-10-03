from typing import Optional

from pydantic import root_validator

from ..tools import OrmBase
from .search_space import SearchSpace


class Document(OrmBase):
    id: Optional[int] = None
    search_space: Optional[SearchSpace] = None
    token: str
    name: str

    @root_validator(pre=True)
    def validate_search_space(cls, values):
        search_space = values.get("search_space")
        if isinstance(search_space, str):
            values["search_space"] = SearchSpace(name=search_space)
        return values
