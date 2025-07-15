from dataclasses import dataclass
from typing import Optional, Tuple
from collections import defaultdict
import numpy as np


@dataclass
class RelativeXY:
    """
    PersonKeypointsからnose を起点とした XY 座標
    """

    nose: Tuple[int, int] = (0, 0)  # 0
    eyeL: Optional[Tuple[int, int]] = None  # 1
    eyeR: Optional[Tuple[int, int]] = None  # 2
    earL: Optional[Tuple[int, int]] = None  # 3
    earR: Optional[Tuple[int, int]] = None  # 4
    shoulderL: Optional[Tuple[int, int]] = None  # 5
    shoulderR: Optional[Tuple[int, int]] = None  # 6
    elbowL: Optional[Tuple[int, int]] = None  # 7
    elbowR: Optional[Tuple[int, int]] = None  # 8
    wristL: Optional[Tuple[int, int]] = None  # 9
    wristR: Optional[Tuple[int, int]] = None  # 10
    hipL: Optional[Tuple[int, int]] = None  # 11
    hipR: Optional[Tuple[int, int]] = None  # 12
    kneeL: Optional[Tuple[int, int]] = None  # 13
    kneeR: Optional[Tuple[int, int]] = None  # 14
    ankleL: Optional[Tuple[int, int]] = None  # 15
    ankleR: Optional[Tuple[int, int]] = None  # 16

    @classmethod
    def from_keypoints(cls, keypoints: "PersonKeypoints") -> "RelativeXY":
        norm_xydict = {}
        scale_factor = 1.0
        if keypoints.is_all_none():
            return cls()

        if (keypoints.nose is not None) & (keypoints.hipL is not None):
            scale_factor = np.linalg.norm(
                np.array(keypoints.nose) - np.array(keypoints.hipL)
            )
        for field in keypoints.__dataclass_fields__:
            point = getattr(keypoints, field)
            if point is None:
                norm_xydict[field] = [0, 0]
            else:
                norm_xydict[field] = [point[0] / scale_factor, point[1] / scale_factor]

        relative_xydict = {"nose": [0, 0]}
        if "nose" not in norm_xydict.keys() or norm_xydict["nose"] == None:
            return RelativeXY(**relative_xydict)
        for key in norm_xydict.keys():
            if key == "nose":
                continue
            point = getattr(keypoints, key)
            if point is None:
                point = [0, 0]
            x = keypoints.nose[0] - point[0]
            y = keypoints.nose[1] - point[1]
            relative_xydict[key] = [x, y]

        return cls(**relative_xydict)

    @classmethod
    def from_mean_relativeXYs(cls, relativeXYs: list["RelativeXY"]) -> "RelativeXY":
        sums = defaultdict(lambda: [0.0, 0.0, 0])  # part_name -> [sum_x, sum_y, count]
        for relativeXY in relativeXYs:
            for field in relativeXY.__dataclass_fields__:
                point = getattr(relativeXY, field)
                if point is not None:
                    sums[field][0] += point[0]
                    sums[field][1] += point[1]
                    sums[field][2] += 1

        avg_data = {}
        for part, (sum_x, sum_y, cnt) in sums.items():
            if cnt > 0:
                avg_data[part] = (sum_x / cnt, sum_y / cnt)
            else:
                avg_data[part] = None

        return cls(**avg_data)

    def flatten_relative_xys(self) -> np.ndarray:
        tmp_data = [
            [val[0], val[1]] if val is not None else [0, 0]
            for val in [
                getattr(self, keypoint) for keypoint in self.__dataclass_fields__
            ]
        ]
        flatten_np = np.array(tmp_data).flatten()
        return flatten_np
