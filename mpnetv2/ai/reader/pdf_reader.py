from pathlib import Path
from typing import List, Optional

from PyPDF2 import PdfReader

from .text_item import TextItem


def pdf_reader(path: Path, search_space: Optional[str] = None) -> List[TextItem]:
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
