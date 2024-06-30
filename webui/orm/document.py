from typing import Optional

from pydantic import root_validator

from ..tools import OrmBase
from .search_space import SearchSpace


class Document(OrmBase):
    id: Optional[int] = None
    token: str
    name: str
    search_space: Optional[SearchSpace] = None
    test_item_length: Optional[int] = None
    language: Optional[str] = None

    @root_validator(pre=True)
    def validate_search_space(cls, values):
        search_space = values.get("search_space")
        text_item_length = values.get("text_item_length")
        if isinstance(search_space, str):
            values["search_space"] = SearchSpace(name=search_space)
        if text_item_length:
            values["text_item_length"] = (int(text_item_length) // 64) * 64
        return values
