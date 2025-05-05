from abc import ABC, abstractmethod
import numpy as np

class BaseNucleiCounter(ABC):
    config: dict

    @abstractmethod
    def count_nuclei(self, image: np.ndarray) -> int:
        pass