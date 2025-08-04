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

        # OpenCVのカラーマップを取得
        self.colormap = getattr(cv2, self.colormap_name, cv2.COLORMAP_HOT)

        self.heatmap_data = np.zeros((self.grid_y, self.grid_x))

    def update(self, centroid_x, centroid_y, image_width, image_height):
        """
        ヒートマップを更新する

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

    def draw(self, image):
        """
        ヒートマップを画像に描画する

        Args:
            image (np.ndarray): 描画対象の画像

        Returns:
            np.ndarray: ヒートマップが描画された画像
        """
        if np.max(self.heatmap_data) == 0:
            return image

        # ヒートマップを正規化（0-255の範囲に）
        normalized_heatmap = (
            self.heatmap_data / np.max(self.heatmap_data) * 255
        ).astype(np.uint8)

        # カラーマップを適用
        heatmap_colored = cv2.applyColorMap(normalized_heatmap, self.colormap)

        # ヒートマップを画像サイズにリサイズ
        heatmap_resized = cv2.resize(
            heatmap_colored,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # 元の画像とヒートマップをブレンド
        blended = cv2.addWeighted(image, 1 - self.alpha, heatmap_resized, self.alpha, 0)

        return blended

    def reset(self):
        """ヒートマップデータをリセットする"""
        self.heatmap_data = np.zeros((self.grid_y, self.grid_x))

    def get_heatmap_data(self):
        """現在のヒートマップデータを取得する"""
        return self.heatmap_data.copy()

    def set_alpha(self, alpha):
        """透明度を設定する"""
        self.alpha = max(0.0, min(1.0, alpha))
