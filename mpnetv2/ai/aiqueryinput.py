from typing import List

from daemon import AiZeroInput

from .reader import TextItem


class AiQueryInput(AiZeroInput):
    def prepare(self) -> List[TextItem]:
        return [TextItem(self.payload["text"])]
