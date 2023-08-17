from pathlib import Path
from typing import List, Optional

from PyPDF2 import PdfFileReader

from .text_item import TextItem


def pdf_reader(path: Path, search_space: Optional[str] = None) -> List[TextItem]:
    reader = PdfFileReader(path)
    return [
        TextItem(
            text=reader.getPage(page_index).extractText(),
            search_space=search_space or "",
            page=page_index,
            paragraph=None,
            path=path,
        )
        for page_index in range(reader.getNumPages())
    ]
