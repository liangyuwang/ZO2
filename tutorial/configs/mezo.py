
from dataclasses import dataclass

from .base import MezoConfig

@dataclass
class GPT2smallMezoConfig(MezoConfig):
    ...

@dataclass
class GPT2mediumMezoConfig(MezoConfig):
    ...

@dataclass
class GPT2largeMezoConfig(MezoConfig):
    ...

@dataclass
class GPT2xlMezoConfig(MezoConfig):
    ...

@dataclass
class OPT125mMezoConfig(MezoConfig):
    ...

@dataclass
class OPT350mMezoConfig(MezoConfig):
    ...

@dataclass
class OPT1_3bMezoConfig(MezoConfig):
    ...

@dataclass
class OPT2_7bMezoConfig(MezoConfig):
    ...

@dataclass
class OPT6_7bMezoConfig(MezoConfig):
    ...

@dataclass
class OPT13bMezoConfig(MezoConfig):
    ...

@dataclass
class OPT30bMezoConfig(MezoConfig):
    ...

@dataclass
class OPT66bMezoConfig(MezoConfig):
    ...

@dataclass
class OPT175bMezoConfig(MezoConfig):
    ...