from typing import Any, Optional


class AiInput:
    @staticmethod
    def get_input() -> Any:
        """
        Get the next model input.
        :param batch_size: The number of files to get.
        :return: The next batch of files.
        """
        raise NotImplementedError("You must implement get_input_batch(batch_size: int) -> Any")

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the input.
        """
        raise NotImplementedError("You must implement __init__(self, ...) -> None")

    def prepare(self) -> Any:
        raise NotImplementedError("You must implement prepare() -> Any")

    def serve(self, inference_data: Optional[Any]) -> None:
        raise NotImplementedError("You must implement serve(inference_data: Any)")
