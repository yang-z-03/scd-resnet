
from enum import Enum
import contextlib
from tqdm import tqdm
import sys

# utility for python ansi logs.

class ConsoleForeground(Enum):
    Black     = '\033[30m'
    Red       = '\033[31m'
    Green     = '\033[32m'
    Yellow    = '\033[33m'
    Blue      = '\033[34m'
    Purple    = '\033[35m'
    DarkGreen = '\033[36m'
    White     = '\033[37m'

class ConsoleBackground(Enum):
    Black     = '\033[40m'
    Red       = '\033[41m'
    Green     = '\033[42m'
    Yellow    = '\033[43m'
    Blue      = '\033[44m'
    Purple    = '\033[45m'
    DarkGreen = '\033[46m'
    White     = '\033[47m'

class ConsoleForegroundOverBlack(Enum):
    Black     = '\033[90m'
    Red       = '\033[91m'
    Green     = '\033[92m'
    Yellow    = '\033[93m'
    Blue      = '\033[94m'
    Purple    = '\033[95m'
    DarkGreen = '\033[96m'
    White     = '\033[97m'

class Logger:
    
    CLEARSTYLES = '\033[0m'

    @staticmethod
    def info(msg):
        print(ConsoleForeground.Blue.value + msg + Logger.CLEARSTYLES)

    @staticmethod
    def infoGreen(msg):
        print(ConsoleForeground.Green.value + msg + Logger.CLEARSTYLES)

    @staticmethod
    def warn(msg):
        print(ConsoleForeground.Yellow.value + msg + Logger.CLEARSTYLES)
    
    @staticmethod
    def err(msg):
        print(ConsoleForeground.Red.value + msg + Logger.CLEARSTYLES)
    
    @staticmethod
    def log(msg):
        print(Logger.CLEARSTYLES + msg)

class TqdmFile(object):
    textIO = None
    def __init__(self, textIO):
        self.textIO = textIO
    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.write(x, file = self.textIO)

@contextlib.contextmanager
def monitorStdOutStream():
    saveStdOut = sys.stdout
    try:
        sys.stdout = TqdmFile(sys.stdout)
        yield saveStdOut
    except Exception as exc:
        raise exc
    finally:
        sys.stdout = saveStdOut
