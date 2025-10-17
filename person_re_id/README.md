# YOLO Multi-Model with Configuration

モジュラー設計されたYOLO検出・トラッキング・ヒートマップシステム

## 特徴

- **ByteTrack + BotSORT**: 複雑なトラッキングアルゴリズム
- **ヒートマップ可視化**: 複数の方式で人の通行パターンを可視化
  - **グリッドベース**: 8x6グリッドで通行頻度を表示
  - **軌跡ベース**: 実際の軌跡を15px線で描画し、最も通られる経路を可視化
- **設定ファイル対応**: YAMLファイルでパラメータを外部化
- **モジュラー設計**: 各機能が独立したクラス

## ファイル構成

```
├── config.yaml              # 設定ファイル
├── config_loader.py         # 設定読み込みクラス
├── class_based_heatmap_visualizer.py # クラス別ヒートマップ可視化クラス
├── heatmap_visualizer.py    # ヒートマップ可視化クラス
├── reid_tracker.py          # Re-IDトラッカークラス
├── yolo_pipeline_processor.py # YOLOパイプライン処理クラス
├── yolo_model_base.py       # YOLOモデル基底クラス
├── yolo_multi_model_refactored.py  # メインスクリプト
├── requirements.txt         # 依存関係
├── bytetrack.yaml           # ByteTrackトラッカー設定（要配置）
├── botsort.yaml            # BotSORTトラッカー設定（要配置）
└── notebooks/              # Jupyter ノートブック
    ├── train_adult_child/   # Adult/Child分類モデル学習
    └── generate_heatmap/    # Google Colabでのヒートマップ生成
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
# 標準モデルでトラッキング
python yolo_multi_model_refactored.py --source video.mp4 --track

# 標準モデルでトラッキング + カウント
python yolo_multi_model_refactored.py --source video.mp4 --track --count

# カスタム設定ファイルを使用
python yolo_multi_model_refactored.py --source video.mp4 --config custom_config.yaml --track

# カスタムモデル（Adult/Child分類）を使用
python yolo_multi_model_refactored.py --source video.mp4 --track --model "best.pt" --dataset-yaml "adult_child.yaml" --classes "0,1" --tracker reid
```

### 出力ファイル

トラッキングを有効にすると、以下のファイルが動画名に基づいたフォルダに生成されます：

```
output/video_name/
├── video_name_output.mp4                    # 処理済み動画（軌跡付き）
├── video_name_labels.txt                    # 検出・トラッキング結果
├── video_name_heatmap_person_class0.jpg     # クラス別ヒートマップ
├── video_name_trajectory_person_class0.jpg  # クラス別軌跡ヒートマップ
├── video_name_heatmap_all_persons.jpg       # 統合ヒートマップ
├── video_name_trajectory_all_persons.jpg    # 統合軌跡ヒートマップ
├── statistics.json                          # 統計情報（JSON）
├── detection_stats.csv                      # 検出統計（CSV）
└── heatmap_stats.csv                        # ヒートマップ統計（CSV）
```

- **クラス別ヒートマップ**: 各クラス（person, car等）ごとの通行パターン
- **軌跡ヒートマップ**: 実際の移動軌跡を線で描画し、重複する経路ほど色が濃くなる
- **統合ヒートマップ**: 全対象クラスを統合した通行パターン
- **統計ファイル**: 検出数、トラック数などの詳細統計

### Google Colabでの使用

#### 手動実行の場合

Google Colabで使用する場合、トラッカー設定ファイルを適切な場所に配置する必要があります：

```bash
# トラッカー設定ファイルをultralyticsのディレクトリにコピー
!cp bytetrack.yaml /usr/local/lib/python*/dist-packages/ultralytics/cfg/trackers/
!cp botsort.yaml /usr/local/lib/python*/dist-packages/ultralytics/cfg/trackers/
!cp bot_sort.py /usr/local/lib/python*/dist-packages/ultralytics/trackers/bot_sort.py


# 実行
!python yolo_multi_model_refactored.py \
  --source video.mp4 \
  --track \
  --model "modelname.pt" \
  --dataset-yaml "adult_child.yaml" \
  --classes "0,1"
```

#### 自動実行ノートブック

より簡単な方法として、`notebooks/generate_heatmap/generate_heatmap.ipynb`を使用できます：

1. ノートブックをGoogle Colabで開く
2. 動画ファイルをアップロード
3. セルを順番に実行（基本的にEnterキーを押すだけ）
4. 自動的にGitHubからコードをクローンし、ヒートマップを生成

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
  use_custom_colormap: false # カスタムカラーマップを使用するか
  custom_colormap_type: "blue_to_red"  # カスタムカラーマップの種類
  generate_simple: true     # 通行回数ベースヒートマップの生成
  generate_weighted: true   # 人数重み付きヒートマップの生成
  generate_trajectory: true # 軌跡ベースヒートマップの生成
  trajectory_thickness: 15  # 軌跡の線の太さ（ピクセル）
  trajectory_alpha: 0.4     # 軌跡ヒートマップの透明度
  trajectory_intensity: 50  # 各軌跡の描画強度（重複すると加算）
  
  # グリッドヒートマップの格子設定
  draw_grid_lines: true     # グリッドの境界線を描画するか
  grid_line_color: [128, 128, 128]  # グリッド線の色 [B, G, R]
  grid_line_thickness: 1    # グリッド線の太さ
```

**カスタムカラーマップの種類:**
- `blue_to_red`: 青（低値）から赤（高値）へのグラデーション
- `green_to_red`: 緑（低値）から赤（高値）へのグラデーション  
- `grayscale`: グレースケール
- `rainbow`: レインボー（HSVベース）

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
from yolo_pipeline_processor import YOLOPipelineProcessor
from config_loader import ConfigLoader

# カスタム設定で初期化
processor = YOLOPipelineProcessor("custom_config.yaml")

# フレーム処理
processed_frame, detections = processor.process_frame(frame, track=True, use_heatmap=True)

# カスタムカラーマップの使用例
# config.yamlで以下を設定:
# heatmap:
#   use_custom_colormap: true
#   custom_colormap_type: "blue_to_red"  # 青から赤のグラデーション
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
- `heatmap.alpha`: グリッドヒートマップの透明度 (0.0-1.0)
- `heatmap.colormap`: OpenCVカラーマップ名
- `heatmap.use_custom_colormap`: カスタムカラーマップを使用するか (true/false)
- `heatmap.custom_colormap_type`: カスタムカラーマップの種類
- `heatmap.generate_simple`: 通行回数ベースヒートマップの生成 (true/false)
- `heatmap.generate_weighted`: 人数重み付きヒートマップの生成 (true/false)
- `heatmap.generate_trajectory`: 軌跡ベースヒートマップの生成 (true/false)
- `heatmap.trajectory_thickness`: 軌跡線の太さ（ピクセル）
- `heatmap.trajectory_alpha`: 軌跡ヒートマップの透明度 (0.0-1.0)
- `heatmap.trajectory_intensity`: 各軌跡の描画強度（0-255、重複すると加算され色が濃くなる）
- `heatmap.draw_grid_lines`: グリッドの境界線を描画するか (true/false)
- `heatmap.grid_line_color`: グリッド線の色 [B, G, R]
- `heatmap.grid_line_thickness`: グリッド線の太さ（ピクセル）

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
# クラス別ヒートマップのみ使用
from class_based_heatmap_visualizer import ClassBasedHeatmapVisualizer
heatmap = ClassBasedHeatmapVisualizer()

# Re-IDトラッカーのみ使用
from reid_tracker import ReIDTracker
tracker = ReIDTracker()

# YOLOパイプライン処理のみ使用
from yolo_pipeline_processor import YOLOPipelineProcessor
processor = YOLOPipelineProcessor()
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