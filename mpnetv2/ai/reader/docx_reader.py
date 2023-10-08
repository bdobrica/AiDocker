from pathlib import Path
from typing import List, Optional

from docx import Document

from .text_item import TextItem


def docx_reader(path: Path, search_space: Optional[str] = None) -> List[TextItem]:
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
