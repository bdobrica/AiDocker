"""
The module converts a document into a list of text items for easy processing with machine learning models.

Usage:
```python
from pathlib import Path
from mpnetv2.ai.reader import read_text

path = Path("path/to/document.pdf")

# Read the document and split it into text items

for item in read_text(path, search_space="en"):
    print(item.text)
```
"""

from pathlib import Path
from typing import List, Optional

from .docx_reader import docx_reader
from .pdf_reader import pdf_reader
from .text_item import TextItem

READERS = {
    ".pdf": pdf_reader,
    ".docx": docx_reader,
}


def read_text(
    path: Path, search_space: Optional[str] = None, max_length: int = 512, overlap: int = 0
) -> List[TextItem]:
    """
    :param path: Path to the document, e.g. "path/to/document.pdf"
    :param search_space: Documents can be grouped into different search spaces, e.g. "en" or "de"
    :param max_length: Maximum length of a text item content, usually matches the maximum input length of the model.
    :param overlap: Overlap between the text content of two consecutive text items. Default: 0
    :return: List of text items
    """
    items = [item.split(max_length=max_length, overlap=overlap) for item in READERS[path.suffix](path, search_space)]
    return [item for sublist in items for item in sublist]
