BASE_PATH = "/content/"
POSES = ["look_forward", "writing_note"]

### video info
VIDEO_PATH = "/content/121教室-20250305-110833_start30sec.mp4"

### video の読み込む範囲
# すべて検出する場合
# DETECT_AREA = None
# MASK_AREA = None
# 特定の位置にする場合
DETECT_AREA = {"min": [400, 0], "max": [1470, 470]}
MASK_AREA = {"min": [1130, 270], "max": [1920, 1080]}

### model info
YOLO_MODEL = "yolov8x-pose.pt"

#### parameter
# ポーズ類似度の threshold、これを下回るとスコアを赤色表示かつ、どのポーズにも属していないとみなす
POSING_SCORE_THRESHOLD: float = 0.7
# sigma を大きくするとスコアがなだらかになる（集中度の判定が甘くなる）
SIGMA: int = 9
# ユーザの中心位置のずれ。大きければ大きいほど違う人物を同じ人物としてしまう可能性があがる。
THRESHOLD_DIST = 80

# ユーザの基準位置をとる秒数
ID_FETCH_DURATION_SEC: int = 5
# IDを検出するインターバル
ID_FETCH_INTERVAL: float = 0.5

# スコア検出のwindow size秒数
WINDOW_SIZE_SEC: int = 5
