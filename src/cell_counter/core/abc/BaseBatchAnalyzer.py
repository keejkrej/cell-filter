from abc import ABC, abstractmethod
from typing import List
from .BaseSingleAnalyzer import BaseSingleAnalyzer, PatternTracker
from .BaseImageLoader import BaseImageLoader
from .BasePatternCropper import BasePatternCropper
from .BasePatternLocator import PatternLocation

class BaseBatchAnalyzer(ABC):
    config: dict
    analyzer: BaseSingleAnalyzer
    image_loader: BaseImageLoader
    pattern_cropper: BasePatternCropper
    pattern_locations: List[PatternLocation]

    @abstractmethod
    def analyze(self, batch: List[int]) -> List[PatternTracker]:
        pass
