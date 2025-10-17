"""
クラス別ヒートマップビジュアライザー
各クラスごとに軌跡とヒートマップを分離して管理
HeatmapVisualizerを継承してクラス別機能を追加

使用例:
    # 2クラス（person=0, car=2）のみを対象とする場合
    visualizer = ClassBasedHeatmapVisualizer(target_classes=[0, 2])

    # 全クラスを対象とする場合
    visualizer = ClassBasedHeatmapVisualizer(target_classes=None)

    # 後から対象クラスを変更する場合
    visualizer.set_target_classes([0, 1, 2])
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
from config_loader import ConfigLoader
from heatmap_visualizer import HeatmapVisualizer


class ClassBasedHeatmapVisualizer(HeatmapVisualizer):
    """クラス別にヒートマップを生成・管理するクラス"""

    def __init__(
        self, config_loader: ConfigLoader = None, target_classes: List[int] = None
    ):
        """
        Args:
            config_loader (ConfigLoader): 設定ローダー
            target_classes (List[int]): 対象とするクラスIDのリスト（Noneの場合は全クラス対象）
        """
        # 親クラスの初期化を呼び出し
        super().__init__(config_loader)

        # 対象クラスを設定（デフォルトは person(0) と car(2) の2クラス）
        self.target_classes = target_classes if target_classes is not None else [0, 2]

        # クラス別ヒートマップデータ
        self.class_heatmap_data = defaultdict(
            lambda: np.zeros((self.grid_y, self.grid_x))
        )
        self.class_weighted_heatmap_data = defaultdict(
            lambda: np.zeros((self.grid_y, self.grid_x))
        )

        # クラス別軌跡データ
        self.class_trajectories = defaultdict(
            lambda: defaultdict(deque)
        )  # {class_id: {track_id: trajectory}}
        self.class_trajectory_images = {}  # {class_id: trajectory_image}

        # 1フレーム目の画像
        self.first_frame = None

        # 軌跡描画設定
        self.trajectory_thickness = self.config.get("heatmap.trajectory_thickness", 15)
        self.trajectory_alpha = self.config.get("heatmap.trajectory_alpha", 0.4)
        self.trajectory_intensity = self.config.get("heatmap.trajectory_intensity", 50)

        # グリッド線の設定
        self.draw_grid_lines = self.config.get("heatmap.draw_grid_lines", True)
        self.grid_line_color = tuple(
            self.config.get("heatmap.grid_line_color", [128, 128, 128])
        )
        self.grid_line_thickness = self.config.get("heatmap.grid_line_thickness", 1)

        # クラス別の色設定
        self.class_colors = self._generate_class_colors()

    def _generate_class_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """クラス別の色を生成"""
        colors = {}
        np.random.seed(42)  # 再現可能な色生成

        # よく使われるクラスID用の固定色
        predefined_colors = {
            0: (255, 0, 0),  # person - 青
            1: (0, 255, 0),  # bicycle - 緑
            2: (0, 0, 255),  # car - 赤
            3: (255, 255, 0),  # motorcycle - シアン
            5: (255, 0, 255),  # bus - マゼンタ
            7: (0, 255, 255),  # truck - 黄色
        }

        colors.update(predefined_colors)

        # その他のクラス用にランダム色を生成
        for i in range(100):  # 最大100クラスまで対応
            if i not in colors:
                colors[i] = tuple(np.random.randint(0, 255, 3).tolist())

        return colors

    def set_first_frame(self, image: np.ndarray):
        """1フレーム目の画像を設定"""
        self.first_frame = image.copy()

    def update_batch(
        self, detections: List[Dict[str, Any]], image_width: int, image_height: int
    ):
        """
        検出結果を一括更新

        Args:
            detections (List[Dict]): 標準化された検出結果
            image_width (int): 画像の幅
            image_height (int): 画像の高さ
        """
        if not detections:
            return

        # クラス別にフレーム内での人数をカウント
        class_frame_counts = defaultdict(lambda: np.zeros((self.grid_y, self.grid_x)))

        for detection in detections:
            class_id = detection["class_id"]

            # 対象クラスでない場合はスキップ
            if class_id not in self.target_classes:
                continue

            track_id = detection.get("track_id")
            bbox = detection["bbox"]

            # 軌跡位置を計算（bbox の [xmedian, ymax]）
            x1, y1, x2, y2 = bbox
            centroid_x = (x1 + x2) / 2  # xmedian
            centroid_y = y2  # ymax（足元）

            # グリッド位置を計算
            grid_x = int((centroid_x / image_width) * self.grid_x)
            grid_y = int((centroid_y / image_height) * self.grid_y)

            # グリッドの範囲内に収める
            grid_x = max(0, min(grid_x, self.grid_x - 1))
            grid_y = max(0, min(grid_y, self.grid_y - 1))

            # クラス別フレーム内人数をカウント
            class_frame_counts[class_id][grid_y, grid_x] += 1

            # 軌跡データを更新（トラッキングIDがある場合）
            if track_id is not None:
                if track_id not in self.class_trajectories[class_id]:
                    self.class_trajectories[class_id][track_id] = deque()
                self.class_trajectories[class_id][track_id].append(
                    (centroid_x, centroid_y)
                )

        # ヒートマップデータを更新
        for class_id, frame_count in class_frame_counts.items():
            # 通行回数ヒートマップ（人が通ったグリッドに+1）
            self.class_heatmap_data[class_id] += (frame_count > 0).astype(float)

            # 人数重み付きヒートマップ（人数に応じて重み付け）
            self.class_weighted_heatmap_data[class_id] += frame_count

    def draw_trajectory_on_heatmap(
        self,
        class_id: int,
        trajectory_points: List[Tuple[float, float]],
        intensity: int = 50,
    ):
        """
        指定クラスの軌跡をヒートマップ画像に描画

        Args:
            class_id (int): クラスID
            trajectory_points (List[Tuple]): 軌跡の点のリスト
            intensity (int): 描画強度
        """
        if self.first_frame is None or len(trajectory_points) < 2:
            return

        # クラス別軌跡画像を初期化
        if class_id not in self.class_trajectory_images:
            self.class_trajectory_images[class_id] = np.zeros_like(self.first_frame)

        # 軌跡を描画
        color = self.class_colors.get(class_id, (255, 255, 255))
        for i in range(1, len(trajectory_points)):
            pt1 = (int(trajectory_points[i - 1][0]), int(trajectory_points[i - 1][1]))
            pt2 = (int(trajectory_points[i][0]), int(trajectory_points[i][1]))

            # 一時的な画像に描画
            temp_img = np.zeros_like(self.class_trajectory_images[class_id])
            cv2.line(temp_img, pt1, pt2, color, self.trajectory_thickness)

            # 加算合成
            self.class_trajectory_images[class_id] = cv2.add(
                self.class_trajectory_images[class_id], temp_img
            )

    def generate_class_heatmap(
        self,
        class_id: int,
        use_weighted: bool = True,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        指定クラスのヒートマップを生成

        Args:
            class_id (int): クラスID
            use_weighted (bool): 重み付きヒートマップを使用するか
            output_path (Optional[str]): 保存パス

        Returns:
            np.ndarray: ヒートマップが重ねられた画像
        """
        if self.first_frame is None:
            raise ValueError("First frame not set. Call set_first_frame() first.")

        # 使用するヒートマップデータを選択
        data_to_use = (
            self.class_weighted_heatmap_data[class_id]
            if use_weighted
            else self.class_heatmap_data[class_id]
        )

        if np.max(data_to_use) == 0:
            print(f"Warning: No heatmap data available for class {class_id}")
            return self.first_frame

        # ヒートマップを生成
        heatmap_image = self._generate_grid_heatmap(data_to_use)

        # 元画像とブレンド
        blended = cv2.addWeighted(
            self.first_frame, 1 - self.alpha, heatmap_image, self.alpha, 0
        )

        # 保存
        if output_path:
            cv2.imwrite(output_path, blended)
            print(f"Class {class_id} heatmap saved to: {output_path}")

        return blended

    def generate_all_class_heatmaps(
        self,
        use_weighted: bool = True,
        output_dir: str = "output",
        file_prefix: str = "",
    ) -> Dict[int, np.ndarray]:
        """
        対象クラスのヒートマップを生成

        Args:
            use_weighted (bool): 重み付きヒートマップを使用するか
            output_dir (str): 出力ディレクトリ
            file_prefix (str): ファイル名のプレフィックス（動画名など）

        Returns:
            Dict[int, np.ndarray]: クラスID別のヒートマップ画像
        """
        heatmaps = {}

        # 対象クラスのみ処理
        for class_id in self.target_classes:
            if class_id in self.class_heatmap_data:
                prefix = f"{file_prefix}_" if file_prefix else ""
                output_path = f"{output_dir}/{prefix}heatmap_class_{class_id}.jpg"
                heatmap = self.generate_class_heatmap(
                    class_id, use_weighted, output_path
                )
                heatmaps[class_id] = heatmap

        return heatmaps

    def generate_combined_heatmap(
        self,
        combine_classes: List[int] = None,
        use_weighted: bool = True,
        output_path: Optional[str] = None,
        label: str = "combined",
    ) -> np.ndarray:
        """
        複数クラスを統合したヒートマップを生成

        Args:
            combine_classes (List[int]): 統合するクラスIDのリスト（Noneの場合は全対象クラス）
            use_weighted (bool): 重み付きヒートマップを使用するか
            output_path (Optional[str]): 保存パス
            label (str): ラベル名

        Returns:
            np.ndarray: 統合ヒートマップが重ねられた画像
        """
        if self.first_frame is None:
            raise ValueError("First frame not set. Call set_first_frame() first.")

        # 統合するクラスを決定
        if combine_classes is None:
            combine_classes = self.target_classes

        # 統合データを初期化
        combined_data = np.zeros((self.grid_y, self.grid_x))

        # 各クラスのデータを統合
        for class_id in combine_classes:
            if class_id in self.class_heatmap_data:
                if use_weighted:
                    combined_data += self.class_weighted_heatmap_data[class_id]
                else:
                    combined_data += self.class_heatmap_data[class_id]

        if np.max(combined_data) == 0:
            print(
                f"Warning: No heatmap data available for combined classes {combine_classes}"
            )
            return self.first_frame

        # ヒートマップを生成
        heatmap_image = self._generate_grid_heatmap(combined_data)

        # 元画像とブレンド
        blended = cv2.addWeighted(
            self.first_frame, 1 - self.alpha, heatmap_image, self.alpha, 0
        )

        # 保存
        if output_path:
            cv2.imwrite(output_path, blended)
            print(f"Combined {label} heatmap saved to: {output_path}")

        return blended

    def generate_combined_trajectory_heatmap(
        self,
        combine_classes: List[int] = None,
        output_path: Optional[str] = None,
        label: str = "combined",
    ) -> np.ndarray:
        """
        複数クラスを統合した軌跡ヒートマップを生成

        Args:
            combine_classes (List[int]): 統合するクラスIDのリスト（Noneの場合は全対象クラス）
            output_path (Optional[str]): 保存パス
            label (str): ラベル名

        Returns:
            np.ndarray: 統合軌跡ヒートマップ画像
        """
        if self.first_frame is None:
            raise ValueError("First frame not set. Call set_first_frame() first.")

        # 統合するクラスを決定
        if combine_classes is None:
            combine_classes = self.target_classes

        # 統合軌跡画像を初期化
        combined_trajectory_image = np.zeros_like(self.first_frame)

        # 各クラスの軌跡を統合軌跡ヒートマップに描画
        for class_id in combine_classes:
            if class_id in self.class_trajectories:
                for track_id, trajectory in self.class_trajectories[class_id].items():
                    if len(trajectory) > 1:
                        self._draw_trajectory_on_combined_image(
                            combined_trajectory_image,
                            list(trajectory),
                            self.trajectory_intensity,
                        )

        # 軌跡画像と1フレーム目をブレンド
        blended = cv2.addWeighted(
            self.first_frame,
            1 - self.trajectory_alpha,
            combined_trajectory_image,
            self.trajectory_alpha,
            0,
        )

        # 保存
        if output_path:
            cv2.imwrite(output_path, blended)
            print(f"Combined {label} trajectory heatmap saved to: {output_path}")

        return blended

    def _draw_trajectory_on_combined_image(
        self,
        target_image: np.ndarray,
        trajectory_points: List[Tuple[float, float]],
        intensity: int = 50,
    ):
        """
        統合軌跡画像に軌跡を描画

        Args:
            target_image (np.ndarray): 描画対象の画像
            trajectory_points (List[Tuple]): 軌跡の点のリスト
            intensity (int): 描画強度
        """
        if len(trajectory_points) < 2:
            return

        # 軌跡を描画（統合用なので単色）
        color = (intensity, intensity, intensity)
        for i in range(1, len(trajectory_points)):
            pt1 = (int(trajectory_points[i - 1][0]), int(trajectory_points[i - 1][1]))
            pt2 = (int(trajectory_points[i][0]), int(trajectory_points[i][1]))

            # 一時的な画像に描画
            temp_img = np.zeros_like(target_image)
            cv2.line(temp_img, pt1, pt2, color, self.trajectory_thickness)

            # 加算合成
            target_image[:] = cv2.add(target_image, temp_img)

    def generate_class_trajectory_heatmap(
        self, class_id: int, output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        指定クラスの軌跡ヒートマップを生成

        Args:
            class_id (int): クラスID
            output_path (Optional[str]): 保存パス

        Returns:
            np.ndarray: 軌跡ヒートマップ画像
        """
        if self.first_frame is None:
            raise ValueError("First frame not set. Call set_first_frame() first.")

        # 全軌跡を軌跡ヒートマップに描画
        for track_id, trajectory in self.class_trajectories[class_id].items():
            if len(trajectory) > 1:
                self.draw_trajectory_on_heatmap(
                    class_id, list(trajectory), self.trajectory_intensity
                )

        # 軌跡画像が存在しない場合は元画像を返す
        if class_id not in self.class_trajectory_images:
            return self.first_frame

        # 軌跡画像と1フレーム目をブレンド
        blended = cv2.addWeighted(
            self.first_frame,
            1 - self.trajectory_alpha,
            self.class_trajectory_images[class_id],
            self.trajectory_alpha,
            0,
        )

        # 保存
        if output_path:
            cv2.imwrite(output_path, blended)
            print(f"Class {class_id} trajectory heatmap saved to: {output_path}")

        return blended

    def _generate_grid_heatmap(self, data: np.ndarray) -> np.ndarray:
        """グリッドベースのヒートマップ画像を生成"""
        h, w = self.first_frame.shape[:2]
        heatmap_colored = np.zeros((h, w, 3), dtype=np.uint8)

        # データを正規化
        normalized_data = (data / np.max(data) * 255).astype(np.uint8)

        # グリッドのセルサイズを計算
        cell_w = w // self.grid_x
        cell_h = h // self.grid_y

        # 各グリッドセルを色で塗りつぶし
        for gy in range(self.grid_y):
            for gx in range(self.grid_x):
                val = normalized_data[gy, gx]

                # カラーマップを使用して色を取得
                if self.use_custom_colormap:
                    color = tuple(int(c) for c in self.colormap[val, 0])
                else:
                    color_map_result = cv2.applyColorMap(
                        np.array([[val]], dtype=np.uint8), self.colormap
                    )
                    color = tuple(int(c) for c in color_map_result[0, 0])

                # グリッドセルの座標を計算
                x1, y1 = gx * cell_w, gy * cell_h
                x2, y2 = (gx + 1) * cell_w, (gy + 1) * cell_h

                # 画像の境界を超えないように調整
                x2 = min(x2, w)
                y2 = min(y2, h)

                # 矩形を塗りつぶし
                cv2.rectangle(heatmap_colored, (x1, y1), (x2, y2), color, -1)

        # グリッド線を描画（オプション）
        if self.draw_grid_lines:
            self._draw_grid_lines(heatmap_colored, cell_w, cell_h, w, h)

        return heatmap_colored

    def _draw_grid_lines(
        self, image: np.ndarray, cell_w: int, cell_h: int, w: int, h: int
    ):
        """グリッド線を描画"""
        # 縦線
        for gx in range(1, self.grid_x):
            x = gx * cell_w
            cv2.line(
                image, (x, 0), (x, h), self.grid_line_color, self.grid_line_thickness
            )

        # 横線
        for gy in range(1, self.grid_y):
            y = gy * cell_h
            cv2.line(
                image, (0, y), (w, y), self.grid_line_color, self.grid_line_thickness
            )

    def set_target_classes(self, target_classes: List[int]):
        """
        対象クラスを設定

        Args:
            target_classes (List[int]): 対象とするクラスIDのリスト
        """
        self.target_classes = target_classes

    def get_target_classes(self) -> List[int]:
        """
        対象クラスを取得

        Returns:
            List[int]: 対象クラスIDのリスト
        """
        return self.target_classes.copy()

    def get_class_statistics(self) -> Dict[int, Dict[str, Any]]:
        """対象クラスの統計情報を取得"""
        stats = {}

        for class_id in self.target_classes:
            if class_id in self.class_heatmap_data:
                total_detections = np.sum(self.class_weighted_heatmap_data[class_id])
                active_grids = np.sum(self.class_heatmap_data[class_id] > 0)
                trajectory_count = len(self.class_trajectories[class_id])

                stats[class_id] = {
                    "total_detections": int(total_detections),
                    "active_grids": int(active_grids),
                    "trajectory_count": trajectory_count,
                    "max_density": float(
                        np.max(self.class_weighted_heatmap_data[class_id])
                    ),
                }

        return stats

    def reset(self):
        """全データをリセット"""
        self.class_heatmap_data.clear()
        self.class_weighted_heatmap_data.clear()
        self.class_trajectories.clear()
        self.class_trajectory_images.clear()
        self.first_frame = None
