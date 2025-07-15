from dataclasses import dataclass
import numpy as np


@dataclass
class DetectXY:
    min: np.ndarray
    max: np.ndarray
