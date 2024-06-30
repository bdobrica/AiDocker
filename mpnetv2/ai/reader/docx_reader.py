from pathlib import Path
from typing import List, Optional

from docx import Document

from .text_item import TextItem


def docx_reader(path: Path, search_space: Optional[str] = None) -> List[TextItem]:
    """
    Uses the python-docx library to read a docx document. The document is split into paragraphs. Each paragraph is
    converted into a text item.
    :param path: Path to the document, e.g. "path/to/document.docx"
    :param search_space: Documents can be grouped into different search spaces, e.g. "en" or "de"
    :return: List of text items
    """
    document = Document(path)
    return [
        TextItem(
            text=paragraph.text,
            token=path.stem,
            search_space=search_space or "",
            page=None,
            paragraph=paragraph_index,
            path=path,
        )
        for paragraph_index, paragraph in enumerate(document.paragraphs)
    ]
