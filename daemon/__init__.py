from .aibatch import AiBatch
from .aibatchdaemon import AiBatchDaemon
from .aiforkdaemon import AiForkDaemon
from .aiinput import AiInput
from .aizerodaemon import AiZeroDaemon
from .aizeroinput import AiZeroInput
from .daemon import Daemon, PathLike
from .filequeuemixin import FileQueueMixin
from .zeroqueuemixin import ZeroQueueMixin

__version__ = "0.9.0"

AiDaemon = AiForkDaemon
AiLiveDaemon = AiZeroDaemon
