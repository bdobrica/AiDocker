from .aibatch import AiBatch
from .aibatchdaemon import AiBatchDaemon
from .aifilebatch import AiFileBatch
from .aifileinput import AiFileInput
from .aiforkdaemon import AiForkDaemon
from .aiinput import AiInput
from .aizerodaemon import AiZeroDaemon
from .aizeroinput import AiZeroInput
from .daemon import Daemon
from .filequeuemixin import FileQueueMixin
from .zeroqueuemixin import ZeroQueueMixin

__version__ = "0.9.0"

AiDaemon = AiForkDaemon
AiLiveDaemon = AiZeroDaemon

__all__ = [
    "AiBatch",
    "AiBatchDaemon",
    "AiFileBatch",
    "AiFileInput",
    "AiForkDaemon",
    "AiInput",
    "AiZeroDaemon",
    "AiZeroInput",
    "Daemon",
    "FileQueueMixin",
    "ZeroQueueMixin",
]
