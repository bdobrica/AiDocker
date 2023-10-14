"""
API callback functions that rely on the file queue.
"""
from .csv import get_csv, put_csv
from .document import put_document
from .image import get_image, put_image
from .metadata import get_json
from .text import put_text
