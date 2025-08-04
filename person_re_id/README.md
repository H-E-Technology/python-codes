# YOLO Multi-Model with Configuration

モジュラー設計されたYOLO検出・トラッキング・ヒートマップシステム

## 特徴

- **ByteTrack + BotSORT**: 複雑なトラッキングアルゴリズム
- **ヒートマップ可視化**: 人の通行頻度を8x6グリッドで表示
- **設定ファイル対応**: YAMLファイルでパラメータを外部化
- **モジュラー設計**: 各機能が独立したクラス

## ファイル構成

```
├── config.yaml              # 設定ファイル
├── config_loader.py         # 設定読み込みクラス
├── heatmap_visualizer.py    # ヒートマップ可視化クラス
├── reid_tracker.py          # Re-IDトラッカークラス
├── yolo_processor.py        # メインYOLO処理クラス
├── yolo_multi_model_refactored.py  # メインスクリプト
├── requirements.txt         # 依存関係
├── bytetrack.yaml           # ByteTrackトラッカー設定（要配置）
└── botsort.yaml            # BotSORTトラッカー設定（要配置）
```

**重要**: `bytetrack.yaml`と`botsort.yaml`は、Ultralyticsライブラリのトラッカー設定ファイルです。これらのファイルは、Ultralyticsのインストールディレクトリ内の適切な場所に配置する必要があります。

## インストール

### ローカル環境

```bash
pip install -r requirements.txt
```

### Google Colab

```bash
# 必要なパッケージのインストール
!pip install ultralytics>=8.0.0 PyYAML>=6.0

# ファイルをアップロードまたはGitHubからクローン
# 動画ファイルとトラッカー設定ファイル（bytetrack.yaml, botsort.yaml）も必要

# トラッカー設定ファイルの配置
!cp /content/bytetrack.yaml /usr/local/lib/python*/dist-packages/ultralytics/cfg/trackers/
!cp /content/botsort.yaml /usr/local/lib/python*/dist-packages/ultralytics/cfg/trackers/
```

## 使用方法

### 基本的な使用

```bash
# 動画ファイルでトラッキング
python yolo_multi_model_refactored.py --source video.mp4 --track

# 動画ファイルでトラッキング + カウント
python yolo_multi_model_refactored.py --source video.mp4 --track --count

# カスタム設定ファイルを使用
python yolo_multi_model_refactored.py --source video.mp4 --config custom_config.yaml --track
```

### Google Colabでの使用

Google Colabで使用する場合、トラッカー設定ファイルを適切な場所に配置する必要があります：

```bash
# トラッカー設定ファイルをultralyticsのディレクトリにコピー
!cp /content/bytetrack.yaml /usr/local/lib/python*/dist-packages/ultralytics/cfg/trackers/
!cp /content/botsort.yaml /usr/local/lib/python*/dist-packages/ultralytics/cfg/trackers/

# 実行
!python yolo_multi_model_refactored.py --source /content/video.mp4 --track --count
```

**注意**: Google Colabでは、セッションを再起動するたびにトラッカー設定ファイルを再配置する必要があります。

### 設定ファイルのカスタマイズ

`config.yaml`を編集して各種パラメータを調整できます：

#### モデル設定
```yaml
model:
  path: "yolo11n-pose.pt"
  conf_threshold: 0.3
  iou_threshold: 0.4
```

#### トラッキング設定
```yaml
tracking:
  max_lost_frames: 30
  max_hidden_frames: 60
  center_area_ratio: 0.4
  line_thickness: 2
  iou_threshold: 0.3
  distance_threshold: 100
```

#### ヒートマップ設定
```yaml
heatmap:
  grid_x: 8
  grid_y: 6
  alpha: 0.6
  colormap: "COLORMAP_HOT"
```

#### 可視化設定
```yaml
visualization:
  colors_count: 80
  font_scale: 0.5
  bbox_thickness: 2
  bbox_color: [0, 0, 225]
  label_text_color: [255, 255, 255]
```

## プログラムでの使用

```python
from yolo_processor import YOLOProcessor
from config_loader import ConfigLoader

# カスタム設定で初期化
config = ConfigLoader("custom_config.yaml")
processor = YOLOProcessor("custom_config.yaml")

# フレーム処理
processed_frame, detections = processor.process_frame(frame, track=True, use_heatmap=True)
```

## 設定可能なパラメータ

### モデル関連
- `model.path`: YOLOモデルファイルのパス
- `model.conf_threshold`: 信頼度閾値 (0.0-1.0)
- `model.iou_threshold`: IoU閾値 (0.0-1.0)

### トラッキング関連
- `tracking.max_lost_frames`: トラック失効までのフレーム数
- `tracking.max_hidden_frames`: 隠れ状態の最大フレーム数
- `tracking.center_area_ratio`: 画面中央領域の比率
- `tracking.iou_threshold`: マッチング用IoU閾値
- `tracking.distance_threshold`: マッチング用距離閾値

### ヒートマップ関連
- `heatmap.grid_x`: 横方向グリッド数
- `heatmap.grid_y`: 縦方向グリッド数
- `heatmap.alpha`: 透明度 (0.0-1.0)
- `heatmap.colormap`: OpenCVカラーマップ名

### 可視化関連
- `visualization.colors_count`: 使用色数
- `visualization.font_scale`: フォントサイズ
- `visualization.bbox_thickness`: バウンディングボックス線の太さ
- `visualization.bbox_color`: バウンディングボックスの色 [B, G, R]

### 出力関連
- `output.fps`: 出力動画のFPS
- `output.codec`: 動画コーデック
- `output.create_output_dir`: 出力ディレクトリの自動作成

## 拡張性

各クラスは独立しているため、他のYOLO識別システムと組み合わせて使用できます：

```python
# ヒートマップのみ使用
from heatmap_visualizer import HeatmapVisualizer
heatmap = HeatmapVisualizer()

# Re-IDトラッカーのみ使用
from reid_tracker import ReIDTracker
tracker = ReIDTracker()
```

## トラブルシューティング

### トラッカーファイルが見つからないエラー

```
FileNotFoundError: [Errno 2] No such file or directory: 'bytetrack.yaml'
```

**解決方法**: トラッカー設定ファイルが正しく配置されていません。

```bash
# Google Colabの場合
!cp /content/bytetrack.yaml /usr/local/lib/python*/dist-packages/ultralytics/cfg/trackers/
!cp /content/botsort.yaml /usr/local/lib/python*/dist-packages/ultralytics/cfg/trackers/

# ローカル環境の場合（Pythonのインストールパスを確認）
cp bytetrack.yaml $(python -c "import ultralytics; print(ultralytics.__path__[0])")/cfg/trackers/
cp botsort.yaml $(python -c "import ultralytics; print(ultralytics.__path__[0])")/cfg/trackers/
```

### 設定ファイルエラー

```
ValueError: Invalid configuration
```

**解決方法**: `config.yaml`の設定値を確認してください。特に以下の値が正しい範囲内にあることを確認：
- `model.conf_threshold`: 0.0-1.0
- `model.iou_threshold`: 0.0-1.0  
- `heatmap.alpha`: 0.0-1.0
- `heatmap.grid_x`, `heatmap.grid_y`: 正の整数