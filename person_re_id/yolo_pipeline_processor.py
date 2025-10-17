"""
YOLO パイプライン処理クラス
YOLO検出、ReIDトラッキング、クラス別ヒートマップ生成を統合したパイプライン
"""

import os
import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import pandas as pd

from config_loader import ConfigLoader
from reid_tracker import ReIDTracker
from class_based_heatmap_visualizer import ClassBasedHeatmapVisualizer
from yolo_model_base import create_yolo_model, YOLOModelBase


class YOLOPipelineProcessor:
    """YOLO検出からヒートマップ生成までの統合パイプライン処理クラス"""

    def __init__(
        self,
        config_path: str = "config.yaml",
        target_classes: List[int] = None,
        custom_classes: Dict[int, str] = None,
        model_path: str = None,
    ):
        """
        Args:
            config_path (str): 設定ファイルのパス
            target_classes (List[int]): 対象とするクラスIDのリスト
            custom_classes (Dict[int, str]): カスタムクラス名辞書
            model_path (str): カスタムモデルパス（設定ファイルより優先）
        """
        # 設定の読み込み
        self.config = ConfigLoader(config_path)

        if not self.config.validate_config():
            raise ValueError("Invalid configuration")

        # モデルパスの決定（引数が優先）
        final_model_path = model_path or self.config.get("model.path", "yolo11n.pt")

        # YOLOモデルの初期化（yolo_model_baseを使用）
        config_dict = self.config.config  # ConfigLoaderから設定辞書を取得
        self.model = create_yolo_model(final_model_path, config_dict, custom_classes)

        # カスタムクラスの保存
        self.custom_classes = custom_classes

        # クラス名の取得
        self.class_names = self.model.get_class_names()
        self.name_to_id = {name: id for id, name in self.class_names.items()}

        # コンポーネントの初期化
        self.reid_tracker = ReIDTracker(self.config)
        self.heatmap_visualizer = ClassBasedHeatmapVisualizer(
            self.config, target_classes
        )

        # 可視化設定
        colors_count = self.config.get("visualization.colors_count", 80)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(colors_count, 3), dtype="uint8")

        # フレーム情報
        self.frame_width = 0
        self.frame_height = 0
        self.current_frame = 0

        # 統計情報
        self.detection_stats = Counter()
        self.tracking_stats = {}

    def set_frame_size(self, width: int, height: int):
        """フレームサイズを設定"""
        self.frame_width = width
        self.frame_height = height

    def set_first_frame(self, frame: np.ndarray):
        """1フレーム目を設定（ヒートマップ用）"""
        self.heatmap_visualizer.set_first_frame(frame)

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        YOLO検出を実行

        Args:
            frame (np.ndarray): 入力フレーム

        Returns:
            List[Dict]: 標準化された検出結果
        """
        # yolo_model_baseを使用して検出を実行
        results = self.model.predict(frame)
        detections = self.model.extract_detections(results)

        return detections

    def process_frame(
        self,
        frame: np.ndarray,
        track: bool = True,
        use_heatmap: bool = True,
        draw_detections: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        フレームを処理（検出→トラッキング→ヒートマップ更新→描画）

        Args:
            frame (np.ndarray): 入力フレーム
            track (bool): トラッキングを実行するか
            use_heatmap (bool): ヒートマップを更新するか
            draw_detections (bool): 検出結果を描画するか

        Returns:
            Tuple[np.ndarray, List[Dict]]: 処理済みフレームと検出結果
        """
        self.current_frame += 1
        processed_frame = frame.copy()

        # フレームサイズを自動設定（未設定の場合）
        if self.frame_width == 0 or self.frame_height == 0:
            self.frame_height, self.frame_width = frame.shape[:2]

        # YOLO検出
        detections = self.detect_objects(frame)

        # 統計更新
        for detection in detections:
            self.detection_stats[detection["class_name"]] += 1

        # トラッキング（ReIDTracker（botsort + bytetrack）を使用）
        if track and detections:
            # ReIDTrackerを使用してトラッキングIDを付与
            # process_dual_trackingに必要なパラメータを全て渡す
            detections = self.reid_tracker.update_tracks(
                detections=detections,
                frame_number=self.current_frame,
                model=self.model.model,  # YOLOモデルオブジェクト
                image=frame,  # 入力フレーム
                frame_width=self.frame_width,
                frame_height=self.frame_height,
            )

        # ヒートマップ更新
        if use_heatmap and detections:
            self.heatmap_visualizer.update_batch(
                detections, self.frame_width, self.frame_height
            )

        # 描画
        if draw_detections:
            processed_frame = self.draw_detections(processed_frame, detections, track)

        return processed_frame, detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        show_tracks: bool = None,
    ) -> np.ndarray:
        """
        検出結果とトラッキング結果を描画（設定に基づいて制御）

        Args:
            frame (np.ndarray): 描画対象フレーム
            detections (List[Dict]): 検出結果
            show_tracks (bool): トラッキング軌跡を表示するか（Noneの場合は設定から取得）

        Returns:
            np.ndarray: 描画済みフレーム
        """
        # 設定から描画制御を取得
        show_detections = self.config.get("output.video.show_detections", True)
        show_track_ids = self.config.get("output.video.show_track_ids", True)

        if show_tracks is None:
            show_tracks = self.config.get("output.video.show_trajectories", True)

        # 検出結果の描画が無効な場合は元のフレームを返す
        if not show_detections:
            return frame
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            class_name = detection["class_name"]
            confidence = detection["confidence"]
            track_id = detection.get("track_id")

            # 色を決定
            if track_id is not None:
                color = tuple(int(c) for c in self.colors[track_id % len(self.colors)])
            else:
                color = (0, 255, 0)  # 緑色（トラッキングなし）

            # バウンディングボックス描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ラベル作成（設定に基づいてトラックIDを表示）
            if track_id is not None and show_track_ids:
                label = f"{class_name} ID:{track_id} {confidence:.2f}"
            else:
                label = f"{class_name} {confidence:.2f}"

            # ラベル背景
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )

            # ラベルテキスト
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        # トラッキング軌跡を描画
        if show_tracks:
            frame = self.reid_tracker.draw_trajectories(frame)

        return frame

    def generate_heatmaps(self, output_dir: str = "output") -> Dict[int, np.ndarray]:
        """
        クラス別ヒートマップを生成（設定に基づいて制御）

        Args:
            output_dir (str): 出力ディレクトリ

        Returns:
            Dict[int, np.ndarray]: クラス別ヒートマップ画像
        """
        # 設定チェック
        if not self.config.get("output.heatmaps.enable", True):
            print("Heatmap generation is disabled in config")
            return {}

        if not self.config.get("output.heatmaps.individual_classes", True):
            print("Individual class heatmaps are disabled in config")
            return {}

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return self.heatmap_visualizer.generate_all_class_heatmaps(
            use_weighted=True, output_dir=output_dir
        )

    def generate_trajectory_heatmaps(
        self, output_dir: str = "output"
    ) -> Dict[int, np.ndarray]:
        """
        クラス別軌跡ヒートマップを生成（設定に基づいて制御）

        Args:
            output_dir (str): 出力ディレクトリ

        Returns:
            Dict[int, np.ndarray]: クラス別軌跡ヒートマップ画像
        """
        # 設定チェック
        if not self.config.get("output.heatmaps.enable", True):
            print("Heatmap generation is disabled in config")
            return {}

        if not self.config.get("output.heatmaps.trajectory_heatmaps", True):
            print("Trajectory heatmaps are disabled in config")
            return {}

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        trajectory_heatmaps = {}
        for class_id in self.heatmap_visualizer.get_target_classes():
            output_path = f"{output_dir}/trajectory_heatmap_class_{class_id}.jpg"
            heatmap = self.heatmap_visualizer.generate_class_trajectory_heatmap(
                class_id, output_path
            )
            trajectory_heatmaps[class_id] = heatmap

        return trajectory_heatmaps

    def generate_combined_heatmaps(
        self,
        output_dir: str = "output",
        combine_classes: List[int] = None,
        label: str = "all_humans",
    ) -> Dict[str, np.ndarray]:
        """
        統合ヒートマップを生成（設定に基づいて制御）

        Args:
            output_dir (str): 出力ディレクトリ
            combine_classes (List[int]): 統合するクラスIDのリスト（Noneの場合は全対象クラス）
            label (str): ラベル名

        Returns:
            Dict[str, np.ndarray]: 統合ヒートマップ画像
        """
        # 設定チェック
        if not self.config.get("output.heatmaps.enable", True):
            print("Heatmap generation is disabled in config")
            return {}

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        combined_heatmaps = {}

        # 統合ヒートマップ（設定に基づいて制御）
        if self.config.get("output.heatmaps.combined_classes", True):
            heatmap_output_path = f"{output_dir}/heatmap_{label}.jpg"
            combined_heatmap = self.heatmap_visualizer.generate_combined_heatmap(
                combine_classes=combine_classes,
                use_weighted=True,
                output_path=heatmap_output_path,
                label=label,
            )
            combined_heatmaps["heatmap"] = combined_heatmap

        # 統合軌跡ヒートマップ（設定に基づいて制御）
        if self.config.get("output.heatmaps.combined_trajectory", True):
            trajectory_output_path = f"{output_dir}/trajectory_{label}.jpg"
            combined_trajectory = (
                self.heatmap_visualizer.generate_combined_trajectory_heatmap(
                    combine_classes=combine_classes,
                    output_path=trajectory_output_path,
                    label=label,
                )
            )
            combined_heatmaps["trajectory"] = combined_trajectory

        return combined_heatmaps

    def save_labels(
        self, detections: List[Dict[str, Any]], frame_id: int, file_path: str
    ):
        """
        検出結果をラベルファイルに保存

        Args:
            detections (List[Dict]): 検出結果
            frame_id (int): フレームID
            file_path (str): 保存先ファイルパス
        """
        with open(file_path, "a") as f:
            for detection in detections:
                track_id = detection.get("track_id", -1)
                class_id = detection["class_id"]
                x1, y1, x2, y2 = detection["bbox"]

                # YOLO形式で保存（フレームID, クラスID, トラックID, bbox）
                f.write(f"{frame_id} {class_id} {track_id} {x1} {y1} {x2} {y2}\n")

    def get_statistics(self) -> Dict[str, Any]:
        """
        処理統計を取得

        Returns:
            Dict: 統計情報
        """
        stats = {
            "detection_stats": dict(self.detection_stats),
            "total_frames": self.current_frame,
            "heatmap_stats": self.heatmap_visualizer.get_class_statistics(),
            "target_classes": self.heatmap_visualizer.get_target_classes(),
            "class_names": {
                class_id: self.class_names.get(class_id, "unknown")
                for class_id in self.heatmap_visualizer.get_target_classes()
            },
        }
        return stats

    def save_statistics(self, output_dir: str = "output"):
        """
        統計情報を保存（設定に基づいて制御）

        Args:
            output_dir (str): 出力ディレクトリ
        """
        # 設定チェック
        if not self.config.get("output.statistics.enable", True):
            print("Statistics saving is disabled in config")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        stats = self.get_statistics()

        # JSON形式で保存
        if self.config.get("output.statistics.save_json", True):
            import json

            json_path = os.path.join(output_dir, "statistics.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"Statistics saved to {json_path}")

        # CSV形式で保存
        if self.config.get("output.statistics.save_csv", True):
            csv_path = os.path.join(output_dir, "detection_stats.csv")

            # 検出統計をDataFrameに変換
            detection_data = []
            for class_name, count in stats["detection_stats"].items():
                detection_data.append(
                    {
                        "class_name": class_name,
                        "detection_count": count,
                        "frames_processed": stats["total_frames"],
                    }
                )

            df = pd.DataFrame(detection_data)
            df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"Detection statistics saved to {csv_path}")

            # ヒートマップ統計もCSVで保存
            heatmap_csv_path = os.path.join(output_dir, "heatmap_stats.csv")
            heatmap_data = []
            for class_id, class_stats in stats["heatmap_stats"].items():
                heatmap_data.append(
                    {
                        "class_id": class_id,
                        "class_name": stats["class_names"].get(class_id, "unknown"),
                        "total_detections": class_stats.get("total_detections", 0),
                        "unique_tracks": class_stats.get("unique_tracks", 0),
                    }
                )

            heatmap_df = pd.DataFrame(heatmap_data)
            heatmap_df.to_csv(heatmap_csv_path, index=False, encoding="utf-8")
            print(f"Heatmap statistics saved to {heatmap_csv_path}")

    def reset(self):
        """全データをリセット"""
        self.reid_tracker = ReIDTracker(self.config)
        self.heatmap_visualizer.reset()
        self.detection_stats.clear()
        self.current_frame = 0

    def set_target_classes(self, target_classes: List[int]):
        """対象クラスを設定"""
        self.heatmap_visualizer.set_target_classes(target_classes)

    def get_target_classes(self) -> List[int]:
        """対象クラスを取得"""
        return self.heatmap_visualizer.get_target_classes()

    def detect_and_track_with_yolo(
        self, frame: np.ndarray, use_reid_tracker: bool = True
    ) -> List[Dict[str, Any]]:
        """
        検出+トラッキングを実行（ReIDTrackerまたはYOLO内蔵トラッキングを選択可能）

        Args:
            frame (np.ndarray): 入力フレーム
            use_reid_tracker (bool): ReIDTrackerを使用するか（デフォルト: True）

        Returns:
            List[Dict]: トラッキング結果を含む検出結果
        """
        # フレームサイズを自動設定（未設定の場合）
        if self.frame_width == 0 or self.frame_height == 0:
            self.frame_height, self.frame_width = frame.shape[:2]
        if use_reid_tracker:
            # ReIDTrackerを使用（推奨）
            detections = self.detect_objects(frame)
            if detections:
                detections = self.reid_tracker.update_tracks(
                    detections=detections,
                    frame_number=self.current_frame,
                    model=self.model.model,  # YOLOモデルオブジェクト
                    image=frame,  # 入力フレーム
                    frame_width=self.frame_width,
                    frame_height=self.frame_height,
                )
            return detections
        else:
            # YOLO内蔵トラッキングを使用
            results = self.model.track(frame)
            detections = self.model.extract_detections(results)
            return detections

    def get_model_info(self) -> Dict[str, Any]:
        """使用中のモデル情報を取得"""
        model_info = self.model.get_model_info()
        if self.custom_classes:
            model_info["custom_classes"] = self.custom_classes
        return model_info

    @staticmethod
    def load_custom_classes_from_yaml(yaml_path: str) -> Dict[int, str]:
        """
        YAMLファイルからカスタムクラス情報を読み込み

        Args:
            yaml_path (str): データセット用YAMLファイルのパス

        Returns:
            Dict[int, str]: クラスID -> クラス名の辞書
        """
        import yaml

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if "names" in data:
                # names辞書からクラス情報を抽出
                if isinstance(data["names"], dict):
                    return {int(k): str(v) for k, v in data["names"].items()}
                elif isinstance(data["names"], list):
                    return {i: name for i, name in enumerate(data["names"])}

            return {}

        except Exception as e:
            print(f"Error loading custom classes from {yaml_path}: {e}")
            return {}

    @classmethod
    def create_with_custom_dataset(
        cls,
        model_path: str,
        dataset_yaml_path: str,
        config_path: str = "config.yaml",
        target_classes: List[int] = None,
    ):
        """
        カスタムデータセットで学習したモデルを使用してインスタンスを作成

        Args:
            model_path (str): 学習済みモデルのパス
            dataset_yaml_path (str): データセット定義YAMLファイルのパス
            config_path (str): 設定ファイルのパス
            target_classes (List[int]): 対象クラス（Noneの場合は全クラス）

        Returns:
            YOLOPipelineProcessor: 設定済みのインスタンス
        """
        # カスタムクラス情報を読み込み
        custom_classes = cls.load_custom_classes_from_yaml(dataset_yaml_path)

        # 対象クラスが指定されていない場合は、カスタムクラスの全てを対象とする
        if target_classes is None and custom_classes:
            target_classes = list(custom_classes.keys())

        return cls(
            config_path=config_path,
            target_classes=target_classes,
            custom_classes=custom_classes,
            model_path=model_path,
        )

    def generate_all_outputs(
        self,
        output_dir: str = "output",
        combine_classes: List[int] = None,
        label: str = "all_humans",
        video_name: str = None,
    ) -> Dict[str, Any]:
        """
        設定に基づいて全ての出力を生成

        Args:
            output_dir (str): 出力ディレクトリ
            combine_classes (List[int]): 統合するクラスIDのリスト
            label (str): 統合ヒートマップのラベル
            video_name (str): 動画名（ファイル名のプレフィックス用）

        Returns:
            Dict[str, Any]: 生成された出力の情報
        """
        results = {}

        # ヒートマップ出力
        if self.config.get("output.heatmaps.enable", True):
            print("Generating heatmaps...")

            # 個別クラスヒートマップ
            if self.config.get("output.heatmaps.individual_classes", True):
                results["individual_heatmaps"] = self.generate_heatmaps(output_dir)

            # 軌跡ヒートマップ
            if self.config.get("output.heatmaps.trajectory_heatmaps", True):
                results["trajectory_heatmaps"] = self.generate_trajectory_heatmaps(
                    output_dir
                )

            # 統合ヒートマップ
            if self.config.get(
                "output.heatmaps.combined_classes", True
            ) or self.config.get("output.heatmaps.combined_trajectory", True):
                results["combined_heatmaps"] = self.generate_combined_heatmaps(
                    output_dir, combine_classes, label
                )

        # 統計情報保存
        if self.config.get("output.statistics.enable", True):
            print("Saving statistics...")
            self.save_statistics(output_dir)
            results["statistics"] = self.get_statistics()

        print(f"All outputs generated in: {output_dir}")
        return results

    def is_output_enabled(self, output_type: str) -> bool:
        """
        指定された出力タイプが有効かチェック

        Args:
            output_type (str): 出力タイプ（例: "heatmaps", "video", "statistics"）

        Returns:
            bool: 有効かどうか
        """
        return self.config.get(f"output.{output_type}.enable", True)
