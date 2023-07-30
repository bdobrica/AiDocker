from .aibatch import AiBatch
from .aibatchdaemon import AiBatchDaemon
from .aiforkdaemon import AiForkDaemon
from .aiinput import AiInput
from .aizerodaemon import AiZeroDaemon
from .aizeroinput import AiZeroInput
from .daemon import Daemon

__version__ = "0.8.13"

AiDaemon = AiForkDaemon
AiLiveDaemon = AiZeroDaemon
