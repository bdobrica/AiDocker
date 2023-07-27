from pathlib import Path
from typing import List

from PyPDF2 import PdfFileReader

from .text_item import TextItem


def pdf_reader(path: Path) -> List[TextItem]:
    reader = PdfFileReader(path)
    return [
        TextItem(
            text=reader.getPage(page_index).extractText(),
            page=page_index,
            paragraph=None,
            path=path,
        )
        for page_index in range(reader.getNumPages())
    ]
