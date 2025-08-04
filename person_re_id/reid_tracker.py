import cv2
import numpy as np
from collections import deque
from config_loader import ConfigLoader


class ReIDTracker:
    """Re-identification とトラッキング機能を提供するクラス（ByteTrack + BotSORT）"""

    def __init__(self, config_loader: ConfigLoader = None):
        """
        Args:
            config_loader (ConfigLoader): 設定ローダー
        """
        if config_loader is None:
            config_loader = ConfigLoader()

        self.config = config_loader

        # 設定から値を取得
        self.max_lost_frames = self.config.get("tracking.max_lost_frames", 30)
        self.max_hidden_frames = self.config.get("tracking.max_hidden_frames", 60)
        self.center_area_ratio = self.config.get("tracking.center_area_ratio", 0.4)
        self.line_thickness = self.config.get("tracking.line_thickness", 2)
        self.draw_points = self.config.get("tracking.draw_points", False)

        # マッチング閾値
        self.iou_threshold = self.config.get("tracking.iou_threshold", 0.3)
        self.distance_threshold = self.config.get("tracking.distance_threshold", 100)
        self.min_iou_for_distance = self.config.get(
            "tracking.min_iou_for_distance", 0.1
        )

        # トラッキング関連のデータ（元のコードと同じ構造）
        self.tracking_trajectories = {}  # 各IDの軌跡を保存（無制限に保存）
        self.lost_tracks = {}  # 途切れたbytetrackのトラックを保存
        self.track_id_mapping = {}  # botsort IDをbytetrack IDにマッピング
        self.last_seen_frame = {}  # 各IDが最後に見られたフレーム番号
        self.hidden_tracks = {}  # 画面中央で隠れた人物の情報を保存
        self.current_frame = 0

        # 色の設定
        colors_count = self.config.get("visualization.colors_count", 80)
        self.colors = np.random.randint(0, 255, size=(colors_count, 3), dtype="uint8")

    def calculate_iou(self, box1, box2):
        """2つのバウンディングボックス間のIoUを計算"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def calculate_distance(self, centroid1, centroid2):
        """2つの中心点間のユークリッド距離を計算"""
        return np.sqrt(
            (centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2
        )

    def get_bbox_centroid(self, bbox):
        """バウンディングボックスの中心点を計算"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def is_in_center_area(self, centroid, image_width, image_height):
        """点が画面中央領域にあるかチェック"""
        center_x_min = image_width * (0.5 - self.center_area_ratio / 2)
        center_x_max = image_width * (0.5 + self.center_area_ratio / 2)
        center_y_min = image_height * (0.5 - self.center_area_ratio / 2)
        center_y_max = image_height * (0.5 + self.center_area_ratio / 2)

        x, y = centroid
        return (center_x_min <= x <= center_x_max) and (
            center_y_min <= y <= center_y_max
        )

    def find_matching_hidden_track(self, centroid, class_id):
        """新しい検出に対してマッチする隠れたトラックを見つける"""
        best_match_id = None
        best_match_distance = self.distance_threshold

        for hidden_id, hidden_info in list(self.hidden_tracks.items()):
            # 同じクラスのオブジェクトのみ比較
            if hidden_info["class"] != class_id:
                continue

            # 隠れてから一定フレーム数以上経過したものは除外
            frames_hidden = self.current_frame - hidden_info["last_frame"]
            if frames_hidden > self.max_hidden_frames:
                continue

            # 最後の位置との距離を計算
            last_position = hidden_info["last_position"]
            distance = self.calculate_distance(centroid, last_position)

            # 距離が近い場合、マッチング候補とする
            if distance < best_match_distance:
                best_match_id = hidden_id
                best_match_distance = distance

        return best_match_id

    def process_dual_tracking(self, model, image, bboxes, frame_width, frame_height):
        """
        ByteTrackとBotSORTを組み合わせた複雑なトラッキング処理
        元のコードのアルゴリズムを正確に再現
        """
        self.current_frame += 1

        # Primary tracker: ByteTrack
        results_bytetrack = model.track(
            image, verbose=False, device=0, persist=True, tracker="bytetrack.yaml"
        )
        # Secondary tracker: BotSORT for補完
        results_botsort = model.track(
            image.copy(), verbose=False, device=0, persist=False, tracker="botsort.yaml"
        )

        # 現在のフレームで検出されたbytetrackのID
        active_bytetrack_ids = set()
        bytetrack_boxes = {}  # bytetrackのID -> bbox

        # ByteTrackの結果を処理
        for predictions in results_bytetrack:
            if (
                predictions is None
                or predictions.boxes is None
                or predictions.boxes.id is None
            ):
                continue

            for bbox in predictions.boxes:
                for scores, classes, bbox_coords, id_ in zip(
                    bbox.conf, bbox.cls, bbox.xyxy, bbox.id
                ):
                    if id_ is not None:
                        bytetrack_id = int(id_)
                        active_bytetrack_ids.add(bytetrack_id)
                        bytetrack_boxes[bytetrack_id] = bbox_coords.tolist()
                        self.last_seen_frame[bytetrack_id] = self.current_frame

        # 失われたトラックを更新
        for track_id in list(self.last_seen_frame.keys()):
            if track_id not in active_bytetrack_ids:
                frames_lost = self.current_frame - self.last_seen_frame[track_id]
                if frames_lost <= self.max_lost_frames:
                    self.lost_tracks[track_id] = self.last_seen_frame[track_id]

                    # 画面中央で消えた人物を hidden_tracks に保存
                    if (
                        track_id in self.tracking_trajectories
                        and len(self.tracking_trajectories[track_id]) > 0
                    ):
                        last_position = self.tracking_trajectories[track_id][-1]

                        # 画面中央で消えたかチェック
                        if self.is_in_center_area(
                            last_position, frame_width, frame_height
                        ):
                            # クラスIDを取得（最後に検出されたクラス）
                            class_id = None
                            for item in bboxes:
                                if len(item) >= 4 and item[3] == track_id:
                                    class_id = item[2]
                                    break

                            if class_id is not None:
                                # 隠れた人物として記録
                                self.hidden_tracks[track_id] = {
                                    "last_position": last_position,
                                    "last_frame": self.current_frame,
                                    "class": class_id,
                                    "trajectory": self.tracking_trajectories[
                                        track_id
                                    ].copy(),
                                }
                                print(
                                    f"Person with ID {track_id} is hiding in center area"
                                )

                elif track_id in self.lost_tracks:
                    del self.lost_tracks[track_id]
                    # hidden_tracks からも削除（隠れ状態が長すぎる場合）
                    if (
                        track_id in self.hidden_tracks
                        and (
                            self.current_frame
                            - self.hidden_tracks[track_id]["last_frame"]
                        )
                        > self.max_hidden_frames
                    ):
                        del self.hidden_tracks[track_id]

        # BotSORTの結果を処理して、失われたトラックを補完
        botsort_boxes = {}  # botsortのID -> (bbox, class, score)

        for predictions in results_botsort:
            if (
                predictions is None
                or predictions.boxes is None
                or predictions.boxes.id is None
            ):
                continue

            for bbox in predictions.boxes:
                for scores, classes, bbox_coords, id_ in zip(
                    bbox.conf, bbox.cls, bbox.xyxy, bbox.id
                ):
                    if id_ is not None:
                        botsort_id = int(id_)
                        botsort_boxes[botsort_id] = (
                            bbox_coords.tolist(),
                            classes,
                            scores,
                        )

        # 失われたbytetrackのIDとbotsortのIDをマッチング
        for lost_id in list(self.lost_tracks.keys()):
            if lost_id in active_bytetrack_ids:  # すでに再検出された場合
                del self.lost_tracks[lost_id]
                continue

            best_match_id = None
            best_match_iou = self.iou_threshold  # 設定から取得
            best_match_dist = self.distance_threshold  # 設定から取得

            if lost_id in bytetrack_boxes:  # 以前のbboxがある場合
                lost_bbox = bytetrack_boxes[lost_id]
                lost_centroid = self.get_bbox_centroid(lost_bbox)

                for botsort_id, (
                    botsort_bbox,
                    botsort_class,
                    _,
                ) in botsort_boxes.items():
                    # すでにマッピングされているIDはスキップ
                    if botsort_id in self.track_id_mapping.values():
                        continue

                    iou = self.calculate_iou(lost_bbox, botsort_bbox)
                    botsort_centroid = self.get_bbox_centroid(botsort_bbox)
                    dist = self.calculate_distance(lost_centroid, botsort_centroid)

                    # IoUが高いか、距離が近い場合にマッチング（設定値を使用）
                    if (iou > best_match_iou) or (
                        iou > self.min_iou_for_distance and dist < best_match_dist
                    ):
                        best_match_id = botsort_id
                        best_match_iou = iou
                        best_match_dist = dist

            if best_match_id is not None:
                self.track_id_mapping[best_match_id] = lost_id
                del self.lost_tracks[lost_id]

        # BotSORTの結果からByteTrackの結果を補完
        # 失われたIDに対応するBotSORTの検出を追加
        for predictions_botsort in results_botsort:
            if (
                predictions_botsort is None
                or predictions_botsort.boxes is None
                or predictions_botsort.boxes.id is None
            ):
                continue

            for bbox_idx, bbox in enumerate(predictions_botsort.boxes):
                for scores, classes, bbox_coords, id_ in zip(
                    bbox.conf, bbox.cls, bbox.xyxy, bbox.id
                ):
                    if id_ is not None:
                        botsort_id = int(id_)
                        # このBotSORTのIDがByteTrackの失われたIDにマッピングされているか確認
                        if botsort_id in self.track_id_mapping:
                            bytetrack_id = self.track_id_mapping[botsort_id]

                            # 対応するByteTrackの結果を見つける
                            for predictions_bytetrack in results_bytetrack:
                                if predictions_bytetrack is not None:
                                    # IDを置き換えて追加（既存のByteTrackのIDと衝突しないように）
                                    if hasattr(bbox, "id"):
                                        # IDを置き換え
                                        bbox.id[bbox.id == id_] = bytetrack_id

                                        # このオブジェクトが現在のByteTrackの結果に存在しない場合のみ追加
                                        if bytetrack_id not in [
                                            int(b.id[i])
                                            for b in predictions_bytetrack.boxes
                                            if hasattr(b, "id")
                                            for i in range(len(b.id))
                                        ]:
                                            # 既存のboxesに追加
                                            if predictions_bytetrack.boxes is not None:
                                                # ここでBotSORTの検出をByteTrackの結果に追加
                                                # 注: 実際の実装はYOLOの内部構造に依存するため、
                                                # 以下は概念的な実装です
                                                self.last_seen_frame[bytetrack_id] = (
                                                    self.current_frame
                                                )

        # 最終的な結果はByteTrackをベースに、BotSORTで補完したもの
        results = results_bytetrack

        # 不要になった軌跡を削除
        for id_ in list(self.tracking_trajectories.keys()):
            if id_ not in [
                int(bbox.id)
                for predictions in results
                if predictions is not None
                for bbox in predictions.boxes
                if bbox.id is not None
            ]:
                del self.tracking_trajectories[id_]

        return results

    def update_trajectories(self, detections, frame_width, frame_height):
        """
        軌跡を更新する

        Args:
            detections (list): 検出結果のリスト [(bbox, score, class, id), ...]
            frame_width (int): フレームの幅
            frame_height (int): フレームの高さ

        Returns:
            list: 更新された中心点のリスト [(id, centroid_x, centroid_y), ...]
        """
        centroids = []

        for detection in detections:
            bbox_coords, scores, classes, id_ = detection
            if id_ is None:
                continue

            track_id = int(id_)

            # バウンディングボックスから中心点を計算
            xmin, ymin, xmax, ymax = bbox_coords
            centroid_x = (xmin + xmax) / 2
            centroid_y = (ymin + ymax) / 2

            # 軌跡を初期化または更新
            if track_id not in self.tracking_trajectories:
                self.tracking_trajectories[track_id] = (
                    deque()
                )  # maxlenを指定せず、無制限に保存

            self.tracking_trajectories[track_id].append((centroid_x, centroid_y))

            centroids.append((track_id, centroid_x, centroid_y))

        return centroids

    def draw_trajectories(self, image):
        """
        軌跡を画像に描画する

        Args:
            image (np.ndarray): 描画対象の画像

        Returns:
            np.ndarray: 軌跡が描画された画像
        """
        colors_count = self.config.get("visualization.colors_count", 80)

        for id_, trajectory in self.tracking_trajectories.items():
            if len(trajectory) < 2:
                continue

            # IDごとに異なる色を生成
            color_id = id_ % colors_count
            color = tuple([int(c) for c in self.colors[color_id]])

            # 軌跡の線を描画
            for i in range(1, len(trajectory)):
                cv2.line(
                    image,
                    (int(trajectory[i - 1][0]), int(trajectory[i - 1][1])),
                    (int(trajectory[i][0]), int(trajectory[i][1])),
                    color,
                    self.line_thickness,
                )

            # 各ポイントを点として描画
            if self.draw_points:
                for point in trajectory:
                    cv2.circle(image, (int(point[0]), int(point[1])), 3, color, -1)

        return image

    def get_trajectory_count(self):
        """現在のトラック数を取得"""
        return len(self.tracking_trajectories)

    def reset(self):
        """全てのトラッキングデータをリセット"""
        self.tracking_trajectories.clear()
        self.lost_tracks.clear()
        self.track_id_mapping.clear()
        self.last_seen_frame.clear()
        self.hidden_tracks.clear()
        self.current_frame = 0
