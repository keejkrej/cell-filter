from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple
import numpy as np

class BaseImageLoader(ABC):
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        pass

    @property
    @abstractmethod
    def n_frame(self) -> int:
        pass

    @property
    @abstractmethod
    def n_channel(self) -> int:
        pass

    @property
    @abstractmethod
    def n_view(self) -> int:
        pass

    @property
    @abstractmethod
    def channel_names(self) -> List[str]:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def get_frame(self, frame_idx: int, channel_idx: int, view_idx: int) -> np.ndarray:
        pass

    def frames(self, channel_idx: int, view_idx: int) -> Iterator[np.ndarray]:
        for frame_idx in range(self.n_frame):
            yield self.get_frame(frame_idx, channel_idx, view_idx)
    