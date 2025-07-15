from dataclasses import dataclass
import cv2
from typing import Optional
from models.settings.detect_xy import DetectXY
import numpy as np
from config import *


@dataclass
class VideoInfo:
    fps: int
    width: int
    height: int
    fourcc: int
    video_path: str

    @classmethod
    def load(cls, video_path: str) -> "VideoInfo":
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cap.release()  # メタ情報だけ保持する

        return cls(fps, width, height, fourcc, video_path)


@dataclass
class DetectXY:
    min: np.ndarray
    max: np.ndarray


@dataclass
class Settings:
    video_info: VideoInfo
    base_path: str = BASE_PATH
    detect_area: Optional[DetectXY] = None
    mask_area: Optional[DetectXY] = None
    posing_score_threhold: float = POSING_SCORE_THRESHOLD
    sigma: int = SIGMA
    threshold_dist: int = THRESHOLD_DIST
    id_fetch_duration_sec: int = ID_FETCH_DURATION_SEC
    id_fetch_interval: float = ID_FETCH_INTERVAL
    window_size_sec: int = WINDOW_SIZE_SEC
