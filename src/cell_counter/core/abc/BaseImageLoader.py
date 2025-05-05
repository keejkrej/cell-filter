from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple
import numpy as np

class BaseImageLoader(ABC):
    shape: Tuple[int, int]
    n_frame: int
    n_channel: int
    n_view: int
    channel_names: List[str]

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def get_frame(self, frame_idx: int, channel_idx: int, view_idx: int) -> np.ndarray:
        pass

    def frames(self, channel_idx: int, view_idx: int) -> Iterator[np.ndarray]:
        for frame_idx in range(self.n_frame):
            yield self.get_frame(frame_idx, channel_idx, view_idx)
    