# Adult/Child Classification Model Training

このディレクトリには、大人と子供を分類するYOLOモデルの学習用ノートブックとスクリプトが含まれています。

## ファイル構成

```
notebooks/train_adult_child/
├── README.md                 # このファイル
├── child_adult.ipynb        # Google Colab用ノートブック
├── child_adult.py           # Python実行用スクリプト
└── (学習用データセット)      # 以下で説明
```

## 必要なデータセット構造

学習を実行するには、以下の構造でデータセットを準備する必要があります：

### 1. 元データセット（combined_datasets_withokinawa.zip）

```
combined_datasets/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── labels/
│   ├── image001.txt
│   ├── image002.txt
│   └── ...
└── adult_child.yaml
```

### 2. データセット分割後の構造

```
dataset_split/
├── images/
│   ├── train/
│   │   ├── image001.jpg
│   │   └── ...
│   ├── val/
│   │   ├── image050.jpg
│   │   └── ...
│   └── test/
│       ├── image090.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image001.txt
    │   └── ...
    ├── val/
    │   ├── image050.txt
    │   └── ...
    └── test/
        ├── image090.txt
        └── ...
```

## アノテーション形式

YOLOフォーマットのアノテーションファイル（.txt）：
```
class_id x_center y_center width height
```

### クラス定義
- `0`: Child（子供）
- `1`: Adult（大人）

## 実行方法

### Google Colabでの実行

1. `child_adult.ipynb`をGoogle Colabにアップロード
2. GPU環境を有効化（Runtime > Change runtime type > GPU）
3. 必要なデータセットファイルをアップロード：
   - `combined_datasets_withokinawa.zip`
   - `okinawa_202510.zip`（ファインチューニング用）
4. セルを順番に実行

### ローカル環境での実行

1. 必要なライブラリをインストール：
```bash
pip install ultralytics opencv-python matplotlib numpy pandas
```

2. データセットを準備：
```bash
# データセットを解凍
unzip combined_datasets_withokinawa.zip
```

3. Pythonスクリプトを実行：
```bash
python child_adult.py
```

## 学習プロセス

### 1. データ前処理

#### データセット分割
- **Train**: 80%
- **Validation**: 10%
- **Test**: 10%
- **乱数シード**: 42（再現性確保）

### 2. 初期学習

```python
from ultralytics import YOLO

# モデル初期化
model = YOLO("yolo11n.pt")  # 事前学習済みモデルを使用

# 学習実行
model.train(
    data="/path/to/adult_child.yaml",
    epochs=80,
    imgsz=640
)
```

### 3. ファインチューニング

```python
# 学習済みモデルを読み込み
best_model = YOLO("runs/detect/train/weights/best.pt")

# ファインチューニング実行
best_model.train(
    data="/path/to/adult_child.yaml",
    epochs=20,
    lr0=0.0005,  # 学習率を下げる
    imgsz=640
)
```

## 学習パラメータ

### 初期学習
- **Epochs**: 80
- **Image Size**: 640x640
- **Base Model**: YOLO11n
- **Learning Rate**: デフォルト

### ファインチューニング
- **Epochs**: 20
- **Image Size**: 640x640
- **Learning Rate**: 0.0005（低めに設定）
- **Base Model**: 初期学習で得られたbest.pt

## 出力ファイル

学習完了後、以下のファイルが生成されます：

```
runs/detect/train/
├── weights/
│   ├── best.pt          # 最良モデル
│   └── last.pt          # 最終エポックモデル
├── results.png          # 学習結果グラフ
├── confusion_matrix.png # 混同行列
├── labels.jpg           # ラベル分布
└── ...
```

## モデルの使用方法

学習済みモデルを使用してメインシステムで推論を実行：

```bash
# カスタムモデルを使用した推論
python yolo_multi_model_refactored.py \
    --source video.mp4 \
    --track \
    --model "runs/detect/train/weights/best.pt" \
    --dataset-yaml "adult_child.yaml" \
    --classes "0,1" \
    --tracker reid
```

## 出力例

学習済みモデルを使用すると、以下のような出力が生成されます：

```
output/video_name/
├── video_name_output.mp4                    # 処理済み動画
├── video_name_heatmap_Child_class0.jpg      # 子供のヒートマップ
├── video_name_heatmap_Adult_class1.jpg      # 大人のヒートマップ
├── video_name_trajectory_Child_class0.jpg   # 子供の軌跡
├── video_name_trajectory_Adult_class1.jpg   # 大人の軌跡
├── video_name_heatmap_all_humans.jpg        # 統合ヒートマップ
├── video_name_trajectory_all_humans.jpg     # 統合軌跡
└── statistics.json                          # 統計情報
```

## トラブルシューティング

### よくある問題

1. **GPU メモリ不足**
   - バッチサイズを小さくする：`batch=8`
   - 画像サイズを小さくする：`imgsz=416`

2. **データセットパスエラー**
   - `adult_child.yaml`のパスが正しいか確認
   - 相対パスと絶対パスの違いに注意

3. **学習が収束しない**
   - 学習率を調整：`lr0=0.001`
   - エポック数を増やす
   - データ拡張を有効化

### デバッグ用コマンド

```python
# データセット検証
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.val(data="adult_child.yaml")

# 予測テスト
results = model.predict(source="test_image.jpg", save=True)
```

## 参考情報

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLO Dataset Format](https://docs.ultralytics.com/datasets/detect/)
- [Training Tips](https://docs.ultralytics.com/modes/train/)

## 注意事項

- 学習には十分なGPUメモリ（推奨：8GB以上）が必要
- データセットの品質が学習結果に大きく影響します
- 適切なデータ拡張とバランスの取れたクラス分布を心がけてください
- 学習前にデータセットの可視化と検証を行うことを推奨します