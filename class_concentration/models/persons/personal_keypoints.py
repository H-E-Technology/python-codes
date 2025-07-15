from dataclasses import dataclass, fields
from typing import Optional, Tuple
import torch

# keypointの位置毎の名称定義
KEYPOINTS_NAMES = [
    "nose",  # 0
    "eyeL",  # 1
    "eyeR",  # 2
    "earL",  # 3
    "earR",  # 4
    "shoulderL",  # 5
    "shoulderR",  # 6
    "elbowL",  # 7
    "elbowR",  # 8
    "wristL",  # 9
    "wristR",  # 10
    "hipL",  # 11
    "hipR",  # 12
    "kneeL",  # 13
    "kneeR",  # 14
    "ankleL",  # 15
    "ankleR",  # 16
]

@dataclass
class PersonKeypoints:
    """
    result から取得した各座標
    """
    nose: Optional[Tuple[float, float]] = None
    eyeL: Optional[Tuple[float, float]] = None
    eyeR: Optional[Tuple[float, float]] = None
    earL: Optional[Tuple[float, float]] = None
    earR: Optional[Tuple[float, float]] = None
    shoulderL: Optional[Tuple[float, float]] = None
    shoulderR: Optional[Tuple[float, float]] = None
    elbowL: Optional[Tuple[float, float]] = None
    elbowR: Optional[Tuple[float, float]] = None
    wristL: Optional[Tuple[float, float]] = None
    wristR: Optional[Tuple[float, float]] = None
    hipL: Optional[Tuple[float, float]] = None
    hipR: Optional[Tuple[float, float]] = None
    kneeL: Optional[Tuple[float, float]] = None
    kneeR: Optional[Tuple[float, float]] = None
    ankleL: Optional[Tuple[float, float]] = None
    ankleR: Optional[Tuple[float, float]] = None

    def is_all_none(self) -> bool:
      return all(getattr(self, f.name) is None for f in fields(self))

    @classmethod
    def from_yolo_result(cls, yolo_result) -> "PersonKeypoints":
      return cls.from_yolo_keypoints(yolo_result.keypoints)

    @classmethod
    def from_yolo_keypoints(cls, keypoints) -> "PersonKeypoints":
      data = {} # 人ごとの座標が入っている
      for conf, xy in zip(keypoints.conf, keypoints.xy):

        for index, keypoint in enumerate(zip(xy, conf)):
            score = keypoint[1]
            # 鼻を基準とするため、鼻のスコアが 0.5 以下の場合とばす
            if (index == 0) & (score < 0.5):
              # xys.append(xydict)
              break

            x = int(keypoint[0][0])
            y = int(keypoint[0][1])
            # 何も入っていない場合飛ばす
            if x == 0 & y == 0:
              continue
            if torch.cuda.is_available():
              data[KEYPOINTS_NAMES[index]] = [x, y] # ,score.item()]
            else:
              data[KEYPOINTS_NAMES[index]] = [x, y] # ,score]
        return cls(**data)
    
    def get_all_keypoints(self):
        return {
           f.name: value for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        }

