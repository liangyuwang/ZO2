
from dataclasses import dataclass

from .base import OffloadingConfig

@dataclass
class GPT2smallOffloadingConfig(OffloadingConfig):
    ...

@dataclass
class GPT2mediumOffloadingConfig(OffloadingConfig):
    ...

@dataclass
class GPT2largeOffloadingConfig(OffloadingConfig):
    ...

@dataclass
class GPT2xlOffloadingConfig(OffloadingConfig):
    ...

@dataclass
class OPT125mOffloadingConfig(OffloadingConfig):
    ...

@dataclass
class OPT350mOffloadingConfig(OffloadingConfig):
    ...

@dataclass
class OPT1_3bOffloadingConfig(OffloadingConfig):
    ...

@dataclass
class OPT2_7bOffloadingConfig(OffloadingConfig):
    ...

@dataclass
class OPT6_7bOffloadingConfig(OffloadingConfig):
    ...

@dataclass
class OPT13bOffloadingConfig(OffloadingConfig):
    ...

@dataclass
class OPT30bOffloadingConfig(OffloadingConfig):
    ...

@dataclass
class OPT66bOffloadingConfig(OffloadingConfig):
    ...

@dataclass
class OPT175bOffloadingConfig(OffloadingConfig):
    ...