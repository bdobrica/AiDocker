from daemon import AiZeroInput


class AiQueryInput(AiZeroInput):
    def prepare(self) -> str:
        return self.payload["text"]
