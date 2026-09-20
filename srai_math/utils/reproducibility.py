import platform,sys,random
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class EnvironmentInfo:
    python:str
    platform:str
    numpy:str
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
def environment_info():
    return EnvironmentInfo(sys.version.split()[0],platform.platform(),np.__version__)
