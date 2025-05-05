from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field

@dataclass(frozen=True)
class PatternLocation:
    top_left: Tuple[int, int]
    top_right: Tuple[int, int]
    bottom_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    bbox: Tuple[int, int, int, int] = field(init=False)
    
    def __post_init__(self):
        x_coords = [self.top_left[0], self.top_right[0],
                   self.bottom_left[0], self.bottom_right[0]]
        y_coords = [self.top_left[1], self.top_right[1],
                   self.bottom_left[1], self.bottom_right[1]]
        
        object.__setattr__(self, 'bbox', (min(x_coords), max(x_coords), min(y_coords), max(y_coords)))
    
class BasePatternLocator(ABC):
    config: dict
    pattern_locations: List[PatternLocation]

    @abstractmethod
    def locate_pattern(self, image: np.ndarray) -> List[PatternLocation]:
        pass
