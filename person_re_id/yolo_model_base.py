"""
YOLO モデルの抽象化ベースクラス
新しいYOLOモデルを簡単に追加できるように設計
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from ultralytics import YOLO


class YOLOModelBase(ABC):
    """YOLOモデルのベースクラス"""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Args:
            model_path (str): モデルファイルのパス
            config (Dict[str, Any]): 設定辞書
        """
        self.model_path = model_path
        self.config = config
        self.model = None
        self.class_names = {}
        self._initialize_model()

    def _initialize_model(self):
        """モデルを初期化"""
        self.model = YOLO(self.model_path)
        self._configure_model()
        self.class_names = self.model.names

    def _configure_model(self):
        """モデルの設定を適用"""
        self.model.overrides["conf"] = self.config.get("model.conf_threshold", 0.3)
        self.model.overrides["iou"] = self.config.get("model.iou_threshold", 0.4)
        self.model.overrides["agnostic_nms"] = self.config.get(
            "model.agnostic_nms", False
        )
        self.model.overrides["max_det"] = self.config.get("model.max_det", 1000)

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> List[Any]:
        """
        画像に対して予測を実行

        Args:
            image (np.ndarray): 入力画像
            **kwargs: 追加のパラメータ

        Returns:
            List[Any]: 予測結果
        """
        pass

    @abstractmethod
    def track(self, image: np.ndarray, **kwargs) -> List[Any]:
        """
        画像に対してトラッキングを実行

        Args:
            image (np.ndarray): 入力画像
            **kwargs: 追加のパラメータ

        Returns:
            List[Any]: トラッキング結果
        """
        pass

    @abstractmethod
    def extract_detections(self, results: List[Any]) -> List[Dict[str, Any]]:
        """
        予測結果から検出情報を抽出

        Args:
            results (List[Any]): モデルの予測結果

        Returns:
            List[Dict[str, Any]]: 標準化された検出結果
                [
                    {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float,
                        'class_id': int,
                        'class_name': str,
                        'track_id': Optional[int],
                        'additional_data': Dict[str, Any]  # モデル固有のデータ
                    },
                    ...
                ]
        """
        pass

    def get_class_names(self) -> Dict[int, str]:
        """クラス名の辞書を取得"""
        return self.class_names

    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        return {
            "model_path": self.model_path,
            "class_names": self.class_names,
            "model_type": self.__class__.__name__,
        }


class YOLODetectionModel(YOLOModelBase):
    """YOLO検出モデル（通常の物体検出）"""

    def predict(self, image: np.ndarray, **kwargs) -> List[Any]:
        """物体検出を実行"""
        return self.model.predict(image, verbose=False, device=0, **kwargs)

    def track(
        self, image: np.ndarray, tracker: str = "bytetrack.yaml", **kwargs
    ) -> List[Any]:
        """トラッキングを実行"""
        return self.model.track(
            image, verbose=False, device=0, persist=True, tracker=tracker, **kwargs
        )

    def extract_detections(self, results: List[Any]) -> List[Dict[str, Any]]:
        """検出結果を標準化された形式で抽出"""
        detections = []

        for result in results:
            if result is None or result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes.xyxy)):
                bbox = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.class_names.get(class_id, f"class_{class_id}")

                # トラッキングIDがある場合
                track_id = None
                if hasattr(boxes, "id") and boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())

                detection = {
                    "bbox": bbox.tolist(),
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                    "track_id": track_id,
                    "additional_data": {},
                }

                detections.append(detection)

        return detections


class YOLOPoseModel(YOLOModelBase):
    """YOLO姿勢推定モデル"""

    def predict(self, image: np.ndarray, **kwargs) -> List[Any]:
        """姿勢推定を実行"""
        return self.model.predict(image, verbose=False, device=0, **kwargs)

    def track(
        self, image: np.ndarray, tracker: str = "bytetrack.yaml", **kwargs
    ) -> List[Any]:
        """姿勢推定 + トラッキングを実行"""
        return self.model.track(
            image, verbose=False, device=0, persist=True, tracker=tracker, **kwargs
        )

    def extract_detections(self, results: List[Any]) -> List[Dict[str, Any]]:
        """姿勢推定結果を標準化された形式で抽出"""
        detections = []

        for result in results:
            if result is None or result.boxes is None:
                continue

            boxes = result.boxes
            keypoints = getattr(result, "keypoints", None)

            for i in range(len(boxes.xyxy)):
                bbox = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.class_names.get(class_id, f"class_{class_id}")

                # トラッキングID
                track_id = None
                if hasattr(boxes, "id") and boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())

                # キーポイント情報
                keypoint_data = None
                if keypoints is not None and i < len(keypoints.data):
                    keypoint_data = keypoints.data[i].cpu().numpy()

                detection = {
                    "bbox": bbox.tolist(),
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                    "track_id": track_id,
                    "additional_data": {
                        "keypoints": (
                            keypoint_data.tolist()
                            if keypoint_data is not None
                            else None
                        )
                    },
                }

                detections.append(detection)

        return detections


class YOLOSegmentationModel(YOLOModelBase):
    """YOLOセグメンテーションモデル"""

    def predict(self, image: np.ndarray, **kwargs) -> List[Any]:
        """セグメンテーションを実行"""
        return self.model.predict(image, verbose=False, device=0, **kwargs)

    def track(
        self, image: np.ndarray, tracker: str = "bytetrack.yaml", **kwargs
    ) -> List[Any]:
        """セグメンテーション + トラッキングを実行"""
        return self.model.track(
            image, verbose=False, device=0, persist=True, tracker=tracker, **kwargs
        )

    def extract_detections(self, results: List[Any]) -> List[Dict[str, Any]]:
        """セグメンテーション結果を標準化された形式で抽出"""
        detections = []

        for result in results:
            if result is None or result.boxes is None:
                continue

            boxes = result.boxes
            masks = getattr(result, "masks", None)

            for i in range(len(boxes.xyxy)):
                bbox = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.class_names.get(class_id, f"class_{class_id}")

                # トラッキングID
                track_id = None
                if hasattr(boxes, "id") and boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())

                # マスク情報
                mask_data = None
                if masks is not None and i < len(masks.xy):
                    mask_data = masks.xy[i]

                detection = {
                    "bbox": bbox.tolist(),
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                    "track_id": track_id,
                    "additional_data": {
                        "mask": mask_data.tolist() if mask_data is not None else None
                    },
                }

                detections.append(detection)

        return detections


class YOLOCustomModel(YOLOModelBase):
    """カスタムデータセットで学習されたYOLOモデル"""

    def __init__(
        self,
        model_path: str,
        config: Dict[str, Any],
        custom_classes: Dict[int, str] = None,
    ):
        """
        Args:
            model_path (str): モデルファイルのパス
            config (Dict[str, Any]): 設定辞書
            custom_classes (Dict[int, str]): カスタムクラス名辞書
        """
        self.custom_classes = custom_classes
        super().__init__(model_path, config)

        # カスタムクラス名が指定されている場合は上書き
        if self.custom_classes:
            self.class_names = self.custom_classes

    def predict(self, image: np.ndarray, **kwargs) -> List[Any]:
        """カスタムモデルで検出を実行"""
        return self.model.predict(image, verbose=False, device=0, **kwargs)

    def track(
        self, image: np.ndarray, tracker: str = "bytetrack.yaml", **kwargs
    ) -> List[Any]:
        """カスタムモデルでトラッキングを実行"""
        return self.model.track(
            image, verbose=False, device=0, persist=True, tracker=tracker, **kwargs
        )

    def extract_detections(self, results: List[Any]) -> List[Dict[str, Any]]:
        """カスタムモデルの検出結果を標準化された形式で抽出"""
        detections = []

        for result in results:
            if result is None or result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes.xyxy)):
                bbox = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.class_names.get(class_id, f"class_{class_id}")

                # トラッキングID
                track_id = None
                if hasattr(boxes, "id") and boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())

                detection = {
                    "bbox": bbox.tolist(),
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                    "track_id": track_id,
                    "additional_data": {
                        "custom_model": True,
                        "model_path": self.model_path,
                    },
                }

                detections.append(detection)

        return detections


def create_yolo_model(
    model_path: str, config: Dict[str, Any], custom_classes: Dict[int, str] = None
) -> YOLOModelBase:
    """
    モデルパスに基づいて適切なYOLOモデルインスタンスを作成

    Args:
        model_path (str): モデルファイルのパス
        config (Dict[str, Any]): 設定辞書
        custom_classes (Dict[int, str]): カスタムクラス名辞書（カスタムモデル用）

    Returns:
        YOLOModelBase: 適切なYOLOモデルインスタンス
    """
    model_path_lower = model_path.lower()

    # カスタムクラスが指定されている場合はカスタムモデルを使用
    if custom_classes:
        return YOLOCustomModel(model_path, config, custom_classes)

    if "pose" in model_path_lower:
        return YOLOPoseModel(model_path, config)
    elif "seg" in model_path_lower:
        return YOLOSegmentationModel(model_path, config)
    else:
        return YOLODetectionModel(model_path, config)
