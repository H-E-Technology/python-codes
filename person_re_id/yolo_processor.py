import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import Counter
import pandas as pd
from reid_tracker import ReIDTracker
from heatmap_visualizer import HeatmapVisualizer
from config_loader import ConfigLoader


class YOLOProcessor:
    """YOLO検出とトラッキングを統合するメインクラス"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path (str): 設定ファイルのパス
        """
        # 設定の読み込み
        self.config = ConfigLoader(config_path)

        # 設定の妥当性チェック
        if not self.config.validate_config():
            raise ValueError("Invalid configuration")

        # YOLOモデルの初期化
        model_path = self.config.get("model.path", "yolo11n-pose.pt")
        self.model = YOLO(model_path)
        self.model.overrides["conf"] = self.config.get("model.conf_threshold", 0.3)
        self.model.overrides["iou"] = self.config.get("model.iou_threshold", 0.4)
        self.model.overrides["agnostic_nms"] = self.config.get(
            "model.agnostic_nms", False
        )
        self.model.overrides["max_det"] = self.config.get("model.max_det", 1000)

        self.names = self.model.names
        self.names = {value: key for key, value in self.names.items()}

        # 可視化設定
        colors_count = self.config.get("visualization.colors_count", 80)
        self.colors = np.random.randint(0, 255, size=(colors_count, 3), dtype="uint8")

        # コンポーネントの初期化
        self.reid_tracker = ReIDTracker(self.config)
        self.heatmap_visualizer = HeatmapVisualizer(self.config)

        # フレーム情報
        self.frame_width = 0
        self.frame_height = 0
        self.current_frame = 0

    def set_frame_size(self, width, height):
        """フレームサイズを設定"""
        self.frame_width = width
        self.frame_height = height

    def process_frame(self, image, track=True, use_heatmap=True):
        """
        フレームを処理する

        Args:
            image (np.ndarray): 入力画像
            track (bool): トラッキングを行うかどうか
            use_heatmap (bool): ヒートマップを使用するかどうか

        Returns:
            tuple: (処理済み画像, 検出結果のリスト)
        """
        self.current_frame += 1
        bboxes = []

        if track:
            # 複雑なトラッキング（ByteTrack + BotSORT）
            results = self.reid_tracker.process_dual_tracking(
                self.model, image, bboxes, self.frame_width, self.frame_height
            )

            # 検出結果の処理
            for predictions in results:
                if predictions is None or predictions.boxes is None:
                    continue

                if predictions.boxes.id is None:
                    continue

                # バウンディングボックスの処理
                image = self._process_detections(image, predictions, bboxes, track=True)
        else:
            # 通常の検出（トラッキングなし）
            results = self.model.predict(image, verbose=False, device=0)

            for predictions in results:
                if predictions is None or predictions.boxes is None:
                    continue

                # バウンディングボックスの処理
                image = self._process_detections(
                    image, predictions, bboxes, track=False
                )

        # トラッキングとヒートマップの更新
        if track and bboxes:
            # 軌跡の更新
            centroids = self.reid_tracker.update_trajectories(
                bboxes, self.frame_width, self.frame_height
            )

            # ヒートマップの更新
            if use_heatmap:
                for track_id, centroid_x, centroid_y in centroids:
                    self.heatmap_visualizer.update(
                        centroid_x, centroid_y, self.frame_width, self.frame_height
                    )

            # 軌跡の描画
            image = self.reid_tracker.draw_trajectories(image)

            # ヒートマップの描画
            if use_heatmap:
                image = self.heatmap_visualizer.draw(image)

        return image, bboxes

    def _process_detections(self, image, predictions, bboxes, track=True):
        """検出結果を処理して画像に描画"""
        # マスクがある場合の処理
        if predictions.masks is not None:
            for bbox, masks in zip(predictions.boxes, predictions.masks):
                image = self._draw_bbox_and_mask(
                    image, bbox, masks, predictions, bboxes, track
                )
        else:
            # マスクがない場合の処理
            for bbox in predictions.boxes:
                image = self._draw_bbox(image, bbox, predictions, bboxes, track)

        return image

    def _draw_bbox_and_mask(self, image, bbox, masks, predictions, bboxes, track):
        """バウンディングボックスとマスクを描画"""
        if track and bbox.id is not None:
            # トラッキング付きの場合
            for scores, classes, bbox_coords, id_ in zip(
                bbox.conf, bbox.cls, bbox.xyxy, bbox.id
            ):
                image = self._draw_single_detection(
                    image, bbox_coords, scores, classes, id_, predictions, bboxes
                )
        else:
            # トラッキングなしの場合
            for scores, classes, bbox_coords in zip(bbox.conf, bbox.cls, bbox.xyxy):
                image = self._draw_single_detection(
                    image, bbox_coords, scores, classes, None, predictions, bboxes
                )

        # マスクの描画
        for mask in masks.xy:
            polygon = mask
            cv2.polylines(image, [np.int32(polygon)], True, (255, 0, 0), thickness=2)

            color_ = [int(c) for c in self.colors[int(classes)]]
            mask_copy = image.copy()
            cv2.fillPoly(mask_copy, [np.int32(polygon)], color_)
            alpha = 0.5
            blended_image = cv2.addWeighted(image, 1 - alpha, mask_copy, alpha, 0)
            image = blended_image.copy()

        return image

    def _draw_bbox(self, image, bbox, predictions, bboxes, track):
        """バウンディングボックスのみを描画"""
        if track and bbox.id is not None:
            # トラッキング付きの場合
            for scores, classes, bbox_coords, id_ in zip(
                bbox.conf, bbox.cls, bbox.xyxy, bbox.id
            ):
                image = self._draw_single_detection(
                    image, bbox_coords, scores, classes, id_, predictions, bboxes
                )
        else:
            # トラッキングなしの場合
            for scores, classes, bbox_coords in zip(bbox.conf, bbox.cls, bbox.xyxy):
                image = self._draw_single_detection(
                    image, bbox_coords, scores, classes, None, predictions, bboxes
                )

        return image

    def _draw_single_detection(
        self, image, bbox_coords, scores, classes, id_, predictions, bboxes
    ):
        """単一の検出結果を描画"""
        xmin, ymin, xmax, ymax = bbox_coords

        # 設定から描画パラメータを取得
        bbox_color = tuple(self.config.get("visualization.bbox_color", [0, 0, 225]))
        bbox_thickness = self.config.get("visualization.bbox_thickness", 2)
        font_scale = self.config.get("visualization.font_scale", 0.5)
        font_thickness = self.config.get("visualization.font_thickness", 1)
        label_bg_color = tuple(
            self.config.get("visualization.label_background_color", [30, 30, 30])
        )
        label_text_color = tuple(
            self.config.get("visualization.label_text_color", [255, 255, 255])
        )

        # バウンディングボックスを描画
        cv2.rectangle(
            image,
            (int(xmin), int(ymin)),
            (int(xmax), int(ymax)),
            bbox_color,
            bbox_thickness,
        )

        # 検出結果をリストに追加
        if id_ is not None:
            bboxes.append([bbox_coords, scores, classes, id_])
            label = f" ID: {int(id_)} {str(predictions.names[int(classes)])} {str(round(float(scores) * 100, 1))}%"
        else:
            bboxes.append([bbox_coords, scores, classes])
            label = f" {str(predictions.names[int(classes)])} {str(round(float(scores) * 100, 1))}%"

        # ラベルを描画
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 4, font_thickness
        )
        dim, baseline = text_size[0], text_size[1]

        cv2.rectangle(
            image,
            (int(xmin), int(ymin)),
            ((int(xmin) + dim[0] // 3) - 20, int(ymin) - dim[1] + baseline),
            label_bg_color,
            cv2.FILLED,
        )

        cv2.putText(
            image,
            label,
            (int(xmin), int(ymin) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            label_text_color,
            font_thickness,
        )

        return image

    def save_labels(self, bboxes, frame_id, output_path):
        """検出結果をファイルに保存"""
        with open(output_path, "a") as file:
            for item in bboxes:
                if len(item) == 4:
                    bbox_coords, scores, classes, id_ = item
                    line = f"{frame_id} {int(classes)} {int(id_)} {round(float(scores), 3)} {int(bbox_coords[0])} {int(bbox_coords[1])} {int(bbox_coords[2])} {int(bbox_coords[3])} -1 -1 -1 -1\n"
                else:
                    bbox_coords, scores, classes = item
                    line = f"{frame_id} {int(classes)} -1 {round(float(scores), 3)} {int(bbox_coords[0])} {int(bbox_coords[1])} {int(bbox_coords[2])} {int(bbox_coords[3])} -1 -1 -1 -1\n"
                file.write(line)

    def get_tracking_stats(self):
        """トラッキング統計を取得"""
        return {
            "active_tracks": self.reid_tracker.get_trajectory_count(),
            "current_frame": self.current_frame,
        }

    def reset(self):
        """全てのデータをリセット"""
        self.reid_tracker.reset()
        self.heatmap_visualizer.reset()
        self.current_frame = 0
