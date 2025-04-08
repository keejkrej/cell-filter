from typing import List, Union
import numpy as np
from ..abc.BaseCounterABC import BaseCounterABC

class CellposeCounter(BaseCounterABC):
    def __init__(self):
        pass

    def count_nuclei(self, images: Union[np.ndarray, List[np.ndarray]]) -> List[int]:
        pass