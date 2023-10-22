from pathlib import Path
from typing import List, Optional

from PyPDF2 import PdfReader

from .text_item import TextItem


def pdf_reader(path: Path, search_space: Optional[str] = None) -> List[TextItem]:
    """
    Uses the PyPDF2 library to read a pdf document. The document is split into pages. Each page is converted into a
    text item.
    :param path: Path to the document, e.g. "path/to/document.pdf"
    :param search_space: Documents can be grouped into different search spaces, e.g. "en" or "de"
    :return: List of text items
    """
    reader = PdfReader(path)
    return [
        TextItem(
            text=page.extract_text(),
            token=path.stem,
            search_space=search_space or "",
            page=page_index,
            paragraph=None,
            path=path,
        )
        for page_index, page in enumerate(reader.pages)
    ]
