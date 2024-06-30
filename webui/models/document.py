from ..tools.db import OrmBase


class Document(OrmBase):
    path: str
    digest: str
    search_space: str
