from pathlib import Path
from typing import List

from .docx_reader import docx_reader
from .pdf_reader import pdf_reader
from .text_item import TextItem

READERS = {
    ".pdf": pdf_reader,
    ".docx": docx_reader,
}


def read_text(path: Path, max_length: int = 512, overlap: int = 0) -> List[TextItem]:
    items = [item.split(max_length=max_length, overlap=overlap) for item in READERS[path.suffix](path)]
    return [item for sublist in items for item in sublist]
