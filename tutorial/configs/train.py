
from dataclasses import dataclass

from .base import TrainConfig

@dataclass
class GPT2smallTrainConfig(TrainConfig):
    ...

@dataclass
class GPT2mediumTrainConfig(TrainConfig):
    ...

@dataclass
class GPT2largeTrainConfig(TrainConfig):
    ...

@dataclass
class GPT2xlTrainConfig(TrainConfig):
    ...

@dataclass
class OPT125mTrainConfig(TrainConfig):
    ...

@dataclass
class OPT350mTrainConfig(TrainConfig):
    ...

@dataclass
class OPT1_3bTrainConfig(TrainConfig):
    ...

@dataclass
class OPT2_7bTrainConfig(TrainConfig):
    ...

@dataclass
class OPT6_7bTrainConfig(TrainConfig):
    ...

@dataclass
class OPT13bTrainConfig(TrainConfig):
    ...

@dataclass
class OPT30bTrainConfig(TrainConfig):
    ...

@dataclass
class OPT66bTrainConfig(TrainConfig):
    ...

@dataclass
class OPT175bTrainConfig(TrainConfig):
    ...