import os

import keras
import numpy as np

from daemon import AiBatchDaemon as Daemon


class AiDaemon(Daemon):
    def load(self) -> None:
        """
        Load the mirNET model. This method is called once when the daemon is started.
        """
        MODEL_PATH = os.getenv("MODEL_PATH", "/opt/app/mirnet/model.h5")
        self.model = keras.models.load_model(MODEL_PATH, compile=False)

    def queue(self) -> None:
        """
        Checks the input file queue for new files. If anew files are found, they are moved to the source folder, packed
        into batches and sent to the AI model for processing. The results are saved to the prepared folder.
        """
        while input_batch := self.input_type.get_input_batch():
            model_input = self.input_type(input_batch)
            for prepared_input in model_input.prepare(self.batch_size):
                model_output = self.ai(prepared_input)
                model_input.serve(model_output)

    def ai(self, input: np.ndarray) -> np.ndarray:
        """
        This method is called for each batch of input data. The input data is a numpy array of shape (N, C, H, W).
        The method should return a numpy array of shape (N, C, H, W).
        :param input: Input data for the AI model in (N, C, H, W) format.
        :return: Output data produced by the AI model in (N, C, H, W) format.
        """
        input = input.astype("float32") / 255.0
        output = self.model.predict(input)
        output = output * 255.0
        output = output.clip(0, 255)
        output = np.uint32(output)
        return output
