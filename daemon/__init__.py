from .aibatch import AiBatch
from .aibatchdaemon import AiBatchDaemon
from .aiforkdaemon import AiForkDaemon
from .aiinput import AiInput
from .aizmqforkdaemon import AiZMQForkDaemon
from .daemon import Daemon

__version__ = "0.8.13"

AiDaemon = AiForkDaemon
AiLiveDaemon = AiZMQForkDaemon
