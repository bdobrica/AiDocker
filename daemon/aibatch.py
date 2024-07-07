from typing import Any, Iterable, Optional


class AiBatch:

    @staticmethod
    def get_input_batch(batch_size: int) -> Iterable[Any]:
        """
        Get the next batch of input files.
        :param batch_size: The number of files to get.
        :return: The next batch of files.
        """
        raise NotImplementedError("You must implement get_input_batch(batch_size: int) -> Any")

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the batch.
        :param staged_files: The files that are part of the current batch.
        """
        raise NotImplementedError("You must implement __init__(self, ...) -> None")

    def prepare(self) -> Any:
        """
        Prepare the batch for processing. This method is responsible for loading the data from the source files and
        returning it in a format that is suitable for AI processing (probably numpy or pytorch.Tensor).
        :return: The data to be processed.
        """
        raise NotImplementedError("You must implement prepare() -> Any")

    def serve(self, inference_data: Optional[Any]) -> None:
        """
        Get's the model output and prepares it by creating the output files under the prepared folder.
        :param inference_data: The data returned by the AI model.
        """
        raise NotImplementedError("You must implement serve(inference_data: Any)")
