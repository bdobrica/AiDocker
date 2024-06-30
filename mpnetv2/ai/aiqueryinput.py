from typing import List

from daemon import AiZeroInput

from .reader import TextItem


class AiQueryInput(AiZeroInput):
    def prepare(self) -> List[TextItem]:
        """
        Returns a list of text items from the request input. Although the request will get only one text item, it is a
        list because the model expects a list of text items as input
        :return: List of text items with the text from the request input, usually a question (natural language query)
        """
        return [TextItem(text=self.payload["text"], search_space=self.payload.get("search_space", ""))]
