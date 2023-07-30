from daemon import AiZeroInput

from .reader import TextItem


class AiQueryInput(AiZeroInput):
    def prepare(self) -> TextItem:
        return TextItem(self.payload["text"])
