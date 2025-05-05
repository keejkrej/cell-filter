from abc import ABC, abstractmethod
from typing import Iterator, List, Dict
import numpy as np

class PatternTracker(ABC):
    tracked: List[int]
    dropped_zero: List[int]
    dropped_many: List[int]
    saved: Dict[int, List[int]]

    @abstractmethod
    def drop_zero(self, idx: int) -> None:
        pass

    @abstractmethod
    def drop_many(self, idx: int) -> None:
        pass

    @abstractmethod
    def save_frame(self, idx: int, frame_idx: int) -> None:
        pass

    @abstractmethod
    def get_tracked_indices(self) -> List[int]:
        pass

    @abstractmethod
    def get_valid_patterns(self) -> Dict[int, List[int]]:
        pass

class BaseSingleAnalyzer(ABC):
    config: dict
    pattern_tracker: PatternTracker

    @abstractmethod
    def analyze(self, frames: Iterator[np.ndarray]) -> PatternTracker:
        pass
