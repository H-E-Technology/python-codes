import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    """設定ファイルを読み込むクラス"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path (str): 設定ファイルのパス
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        return config

    def get(self, key_path: str, default=None):
        """
        ドット記法でネストした設定値を取得

        Args:
            key_path (str): 設定のキーパス（例: "tracking.max_lost_frames"）
            default: デフォルト値

        Returns:
            設定値またはデフォルト値
        """
        keys = key_path.split(".")
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        設定のセクション全体を取得

        Args:
            section (str): セクション名

        Returns:
            Dict[str, Any]: セクションの設定
        """
        return self.config.get(section, {})

    def update_config(self, key_path: str, value):
        """
        設定値を更新

        Args:
            key_path (str): 設定のキーパス
            value: 新しい値
        """
        keys = key_path.split(".")
        config = self.config

        # 最後のキー以外をたどる
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # 最後のキーに値を設定
        config[keys[-1]] = value

    def save_config(self, output_path: str = None):
        """
        設定をファイルに保存

        Args:
            output_path (str): 出力パス（Noneの場合は元のファイルを上書き）
        """
        output_path = output_path or self.config_path

        with open(output_path, "w", encoding="utf-8") as file:
            yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)

    def reload(self):
        """設定ファイルを再読み込み"""
        self.config = self._load_config()

    def validate_config(self) -> bool:
        """
        設定の妥当性をチェック

        Returns:
            bool: 設定が妥当かどうか
        """
        required_sections = ["model", "tracking", "heatmap", "visualization", "output"]

        for section in required_sections:
            if section not in self.config:
                print(f"Missing required section: {section}")
                return False

        # 数値の範囲チェック
        if not (0.0 <= self.get("model.conf_threshold", 0.3) <= 1.0):
            print("model.conf_threshold must be between 0.0 and 1.0")
            return False

        if not (0.0 <= self.get("model.iou_threshold", 0.4) <= 1.0):
            print("model.iou_threshold must be between 0.0 and 1.0")
            return False

        if not (0.0 <= self.get("heatmap.alpha", 0.6) <= 1.0):
            print("heatmap.alpha must be between 0.0 and 1.0")
            return False

        if self.get("heatmap.grid_x", 8) <= 0 or self.get("heatmap.grid_y", 6) <= 0:
            print("heatmap grid dimensions must be positive")
            return False

        return True
