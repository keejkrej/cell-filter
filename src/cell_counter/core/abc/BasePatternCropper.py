from abc import ABC, abstractmethod
import numpy as np
from .BasePatternLocator import PatternLocation

class BasePatternCropper(ABC):
    config: dict

    @abstractmethod
    def crop(self, image: np.ndarray, pattern_location: PatternLocation) -> np.ndarray:
        pass
