from pathlib import Path
from typing import List

from docx import Document

from .text_item import TextItem


def docx_reader(path: Path) -> List[TextItem]:
    document = Document(path)
    return [
        TextItem(
            text=paragraph.text,
            page=None,
            paragraph=paragraph_index,
            path=path,
        )
        for paragraph_index, paragraph in enumerate(document.paragraphs)
    ]
