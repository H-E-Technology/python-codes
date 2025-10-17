import cv2
import numpy as np
from config_loader import ConfigLoader


class HeatmapVisualizer:
    """人の軌跡をヒートマップとして可視化するクラス"""

    def __init__(self, config_loader: ConfigLoader = None):
        """
        Args:
            config_loader (ConfigLoader): 設定ローダー
        """
        if config_loader is None:
            config_loader = ConfigLoader()

        self.config = config_loader

        # 設定から値を取得
        self.grid_x = self.config.get("heatmap.grid_x", 8)
        self.grid_y = self.config.get("heatmap.grid_y", 6)
        self.alpha = self.config.get("heatmap.alpha", 0.6)
        self.colormap_name = self.config.get("heatmap.colormap", "COLORMAP_HOT")
        self.use_custom_colormap = self.config.get("heatmap.use_custom_colormap", False)

        # カラーマップを設定
        if self.use_custom_colormap:
            self.colormap = self._create_custom_colormap()
        else:
            # OpenCVのカラーマップを取得
            self.colormap = getattr(cv2, self.colormap_name, cv2.COLORMAP_HOT)

        # ヒートマップデータ（通行回数をカウント）
        self.heatmap_data = np.zeros((self.grid_y, self.grid_x))

        # 人数重み付きヒートマップデータ（複数人が同時に通った場所を重視）
        self.weighted_heatmap_data = np.zeros((self.grid_y, self.grid_x))

        # 1フレーム目の画像を保存
        self.first_frame = None

        # 軌跡ベースのヒートマップ用
        self.trajectory_image = None  # 軌跡を描画する黒い画像
        self.trajectory_thickness = self.config.get("heatmap.trajectory_thickness", 5)
        self.trajectory_alpha = self.config.get("heatmap.trajectory_alpha", 0.7)
        self.trajectory_intensity = self.config.get("heatmap.trajectory_intensity", 50)

        # グリッド線の設定
        self.draw_grid_lines = self.config.get("heatmap.draw_grid_lines", True)
        self.grid_line_color = tuple(
            self.config.get("heatmap.grid_line_color", [128, 128, 128])
        )
        self.grid_line_thickness = self.config.get("heatmap.grid_line_thickness", 1)

    def _create_custom_colormap(self):
        """
        カスタムカラーマップを作成する

        Returns:
            np.ndarray: カスタムカラーマップ (256, 1, 3)
        """
        # カスタムカラーマップの設定を取得
        custom_type = self.config.get("heatmap.custom_colormap_type", "blue_to_red")

        custom_map = np.zeros((256, 1, 3), dtype=np.uint8)

        if custom_type == "black_to_red":
            # 黒から深い青、そして真っ赤へのグラデーション
            for i in range(256):
                if i < 64:
                    # 黒から深い青へ (0-63)
                    progress = i / 63.0
                    r = 0
                    g = 0
                    b = int(progress * 80)  # 深い青まで
                elif i < 128:
                    # 深い青から青へ (64-127)
                    progress = (i - 64) / 63.0
                    r = 0
                    g = 0
                    b = int(80 + progress * 175)  # 深い青から明るい青へ
                else:
                    # 青から赤へ (128-255)
                    progress = (i - 128) / 127.0
                    r = int(progress * 255)  # 赤成分を増加
                    g = 0
                    b = int(255 * (1 - progress))  # 青成分を減少
                custom_map[i, 0] = [b, g, r]

        elif custom_type == "blue_to_red":
            # 青から赤へのグラデーション
            for i in range(256):
                r = i  # 高い値ほど赤く
                g = 0  # 緑は使わない
                b = 255 - i  # 低い値ほど青く
                custom_map[i, 0] = [b, g, r]

        elif custom_type == "green_to_red":
            # 緑から赤へのグラデーション
            for i in range(256):
                r = i  # 高い値ほど赤く
                g = 255 - i  # 低い値ほど緑く
                b = 0  # 青は使わない
                custom_map[i, 0] = [b, g, r]

        elif custom_type == "grayscale":
            # グレースケール
            for i in range(256):
                custom_map[i, 0] = [i, i, i]

        elif custom_type == "rainbow":
            # レインボー（HSVベース）
            for i in range(256):
                # HSVで色相を変化させる
                hue = int(i * 180 / 256)  # 0-180の範囲
                hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                custom_map[i, 0] = bgr[0, 0]

        else:
            # デフォルト: 黒から深い青、そして真っ赤
            for i in range(256):
                if i < 64:
                    # 黒から深い青へ
                    progress = i / 63.0
                    r = 0
                    g = 0
                    b = int(progress * 80)
                elif i < 128:
                    # 深い青から青へ
                    progress = (i - 64) / 63.0
                    r = 0
                    g = 0
                    b = int(80 + progress * 175)
                else:
                    # 青から赤へ
                    progress = (i - 128) / 127.0
                    r = int(progress * 255)
                    g = 0
                    b = int(255 * (1 - progress))
                custom_map[i, 0] = [b, g, r]

        return custom_map

    def update_batch(self, centroids_list, image_width, image_height):
        """
        複数の中心点を一度に更新する（フレーム単位）

        Args:
            centroids_list (list): [(track_id, centroid_x, centroid_y), ...] のリスト
            image_width (int): 画像の幅
            image_height (int): 画像の高さ
        """
        if not centroids_list:
            return

        # フレーム内でのグリッド別人数をカウント
        frame_grid_count = np.zeros((self.grid_y, self.grid_x))

        for track_id, centroid_x, centroid_y in centroids_list:
            # 中心点がどのグリッドに属するかを計算
            grid_x = int((centroid_x / image_width) * self.grid_x)
            grid_y = int((centroid_y / image_height) * self.grid_y)

            # グリッドの範囲内に収める
            grid_x = max(0, min(grid_x, self.grid_x - 1))
            grid_y = max(0, min(grid_y, self.grid_y - 1))

            # フレーム内でのグリッド別人数をカウント
            frame_grid_count[grid_y, grid_x] += 1

        # 通行回数ヒートマップを更新（人が通ったグリッドに+1）
        self.heatmap_data += (frame_grid_count > 0).astype(float)

        # 人数重み付きヒートマップを更新（人数に応じて重み付け）
        self.weighted_heatmap_data += frame_grid_count

    def update_single(self, centroid_x, centroid_y, image_width, image_height):
        """
        単一の中心点を更新する（後方互換性のため）

        Args:
            centroid_x (float): 中心点のX座標
            centroid_y (float): 中心点のY座標
            image_width (int): 画像の幅
            image_height (int): 画像の高さ
        """
        # 中心点がどのグリッドに属するかを計算
        grid_x = int((centroid_x / image_width) * self.grid_x)
        grid_y = int((centroid_y / image_height) * self.grid_y)

        # グリッドの範囲内に収める
        grid_x = max(0, min(grid_x, self.grid_x - 1))
        grid_y = max(0, min(grid_y, self.grid_y - 1))

        # ヒートマップデータを更新（カウントを増加）
        self.heatmap_data[grid_y, grid_x] += 1
        self.weighted_heatmap_data[grid_y, grid_x] += 1

    def set_first_frame(self, image):
        """
        1フレーム目の画像を保存する

        Args:
            image (np.ndarray): 1フレーム目の画像
        """
        self.first_frame = image.copy()

        # 軌跡描画用の黒い画像を初期化
        self.trajectory_image = np.zeros_like(image)

    def draw_trajectory_on_heatmap(self, trajectory_points, intensity=50):
        """
        軌跡を軌跡ヒートマップ画像に描画する

        Args:
            trajectory_points (list): 軌跡の点のリスト [(x, y), ...]
            intensity (int): 描画する線の強度（0-255、重複すると加算される）
        """
        if self.trajectory_image is None or len(trajectory_points) < 2:
            return

        # 軌跡を指定した強度で描画（重複すると加算される）
        for i in range(1, len(trajectory_points)):
            pt1 = (int(trajectory_points[i - 1][0]), int(trajectory_points[i - 1][1]))
            pt2 = (int(trajectory_points[i][0]), int(trajectory_points[i][1]))

            # 現在の値を取得して加算（255を超えないように制限）
            temp_img = np.zeros_like(self.trajectory_image)
            cv2.line(
                temp_img,
                pt1,
                pt2,
                (intensity, intensity, intensity),
                self.trajectory_thickness,
            )

            # 加算合成（255を上限とする）
            self.trajectory_image = cv2.add(self.trajectory_image, temp_img)

    def generate_final_heatmap(self, use_weighted=True, output_path=None):
        """
        最終的なヒートマップを生成する

        Args:
            use_weighted (bool): 人数重み付きヒートマップを使用するか
            output_path (str): 保存パス（Noneの場合は保存しない）

        Returns:
            np.ndarray: ヒートマップが重ねられた画像
        """
        if self.first_frame is None:
            raise ValueError("First frame not set. Call set_first_frame() first.")

        # 使用するヒートマップデータを選択
        data_to_use = self.weighted_heatmap_data if use_weighted else self.heatmap_data

        if np.max(data_to_use) == 0:
            print("Warning: No heatmap data available")
            return self.first_frame

        # ヒートマップを正規化（0-255の範囲に）
        normalized_data = (data_to_use / np.max(data_to_use) * 255).astype(np.uint8)

        # 格子状ヒートマップを作成
        h, w = self.first_frame.shape[:2]
        heatmap_colored = np.zeros((h, w, 3), dtype=np.uint8)

        # グリッドのセルサイズを計算
        cell_w = w // self.grid_x
        cell_h = h // self.grid_y

        # 各グリッドセルを色で塗りつぶし
        for gy in range(self.grid_y):
            for gx in range(self.grid_x):
                val = normalized_data[gy, gx]

                # カラーマップを使用して色を取得
                if self.use_custom_colormap:
                    # カスタムカラーマップから直接色を取得
                    color = tuple(int(c) for c in self.colormap[val, 0])
                else:
                    # OpenCVのカラーマップを適用
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
            # 縦線を描画
            for gx in range(1, self.grid_x):
                x = gx * cell_w
                cv2.line(
                    heatmap_colored,
                    (x, 0),
                    (x, h),
                    self.grid_line_color,
                    self.grid_line_thickness,
                )

            # 横線を描画
            for gy in range(1, self.grid_y):
                y = gy * cell_h
                cv2.line(
                    heatmap_colored,
                    (0, y),
                    (w, y),
                    self.grid_line_color,
                    self.grid_line_thickness,
                )

        heatmap_resized = heatmap_colored

        # 元の画像とヒートマップをブレンド
        blended = cv2.addWeighted(
            self.first_frame, 1 - self.alpha, heatmap_resized, self.alpha, 0
        )

        # 保存
        if output_path:
            cv2.imwrite(output_path, blended)
            print(f"Heatmap saved to: {output_path}")

        return blended

    def generate_trajectory_heatmap(self, output_path=None):
        """
        軌跡ベースのヒートマップを生成する

        Args:
            output_path (str): 保存パス（Noneの場合は保存しない）

        Returns:
            np.ndarray: 軌跡ヒートマップが重ねられた画像
        """
        if self.first_frame is None:
            raise ValueError("First frame not set. Call set_first_frame() first.")

        if self.trajectory_image is None:
            print("Warning: No trajectory data available")
            return self.first_frame

        # 軌跡画像と1フレーム目をブレンド
        blended = cv2.addWeighted(
            self.first_frame,
            1 - self.trajectory_alpha,
            self.trajectory_image,
            self.trajectory_alpha,
            0,
        )

        # 保存
        if output_path:
            cv2.imwrite(output_path, blended)
            print(f"Trajectory heatmap saved to: {output_path}")

        return blended

    def draw(self, image):
        """
        ヒートマップを画像に描画する（後方互換性のため残す）

        Args:
            image (np.ndarray): 描画対象の画像

        Returns:
            np.ndarray: 元の画像をそのまま返す（動画には描画しない）
        """
        # 動画には描画しない
        return image

    def reset(self):
        """ヒートマップデータをリセットする"""
        self.heatmap_data = np.zeros((self.grid_y, self.grid_x))
        self.weighted_heatmap_data = np.zeros((self.grid_y, self.grid_x))
        self.first_frame = None
        self.trajectory_image = None

    def get_heatmap_data(self):
        """現在のヒートマップデータを取得する"""
        return self.heatmap_data.copy()

    def get_weighted_heatmap_data(self):
        """人数重み付きヒートマップデータを取得する"""
        return self.weighted_heatmap_data.copy()

    def set_alpha(self, alpha):
        """透明度を設定する"""
        self.alpha = max(0.0, min(1.0, alpha))
