from google.colab.patches import cv2_imshow
import cv2
import glob
import math
import os

from models.persons.tracked_person import TrackedPerson
from models.persons.relative_xy import RelativeXY
from models.persons.personal_keypoints import PersonKeypoints
from models.persons.ideal_pose import IdealPose
from collections import defaultdict, Counter


COLOR_MAP = {
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "orange": (0, 165, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}

# 名前ベースのスケルトン定義
# 描画に利用する
SKELETON_NAME_PAIRS = [
    ("shoulderL", "elbowL"),
    ("elbowL", "wristL"),
    ("shoulderR", "elbowR"),
    ("elbowR", "wristR"),
    ("hipL", "kneeL"),
    ("kneeL", "ankleL"),
    ("hipR", "kneeR"),
    ("kneeR", "ankleR"),
    ("shoulderL", "shoulderR"),
    ("hipL", "hipR"),
    ("shoulderL", "hipL"),
    ("shoulderR", "hipR"),
    ("nose", "eyeL"),
    ("nose", "eyeR"),
    ("eyeL", "earL"),
    ("eyeR", "earR"),
    ("nose", "shoulderL"),
    ("nose", "shoulderR"),
]


def get_ideal_pose_list(poses: list, ideal_pose_dir: str, yolo_model):
    ideal_pose_list = []
    for pose in poses:
        ideal_relative_xy_list: list[RelativeXY] = []
        files = glob.glob(os.path.join(ideal_pose_dir, pose, "*"))

        for file in files:
            capture = cv2.imread(file)
            results = yolo_model(capture, verbose=False)
            # 描画された画像を取得（NumPy配列）
            result_img = results[0].plot()

            # 画像の中に複数人いる場合があるため、リスト形式
            tmp_relative_xy_list: list[RelativeXY] = get_relative_xys(results)
            ideal_relative_xy_list += tmp_relative_xy_list

        ideal_pose = IdealPose.from_xys(
            pose_name=pose, relative_xy_list=ideal_relative_xy_list
        )
        ideal_pose_list.append(ideal_pose)
    return ideal_pose_list


## 理想の姿勢を取得（おそらくここはやり方を変える
## 何かしらよいポーズを 2, 3用意し、それとの類似度あるいは分類問題にする）
def get_relative_xys(results) -> list[RelativeXY]:
    """
    YOLO解析結果から、鼻との相対位置のリストを取得
    そのキャプチャに存在する人のデータがリスト形式になって返却される
    """
    relative_xy_list = []
    for result in results:
        person_keypoints = PersonKeypoints.from_yolo_result(result)

        if person_keypoints.is_all_none():
            continue
        relative_xy = RelativeXY.from_keypoints(person_keypoints)

        relative_xy_list.append(relative_xy)
    return relative_xy_list


def show_detect_persons(frame, filtered_people_xys: dict[int, TrackedPerson]) -> None:
    print("もし抽出箇所が微妙な場合 CTRL + C を 2回押して強制終了")
    for id, tracked_person in filtered_people_xys.items():
        font_color = COLOR_MAP["red"]
        cv2.putText(
            frame,
            f"{id}",
            (tracked_person.center_x, tracked_person.center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            font_color,
            2,
            cv2.LINE_AA,
        )
    scale = 0.5
    resized_img = cv2.resize(frame, None, fx=scale, fy=scale)
    cv2_imshow(resized_img)


def get_avg_dict(target_list):
    # 各キーごとの合計値とカウントを保持する辞書
    sum_dict = defaultdict(float)
    count_dict = defaultdict(int)

    # 合計値とカウントを計算
    for score_map in target_list:
        for key, value in score_map.items():
            sum_dict[key] += value
            count_dict[key] += 1

    # 平均値を計算
    avg_dict = {key: sum_dict[key] / count_dict[key] for key in sum_dict}
    return avg_dict


import subprocess


# utils に入れる
def get_audio_duration(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-i",
            path,
            "-show_entries",
            "format=duration",
            "-v",
            "quiet",
            "-of",
            "csv=p=0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return float(result.stdout.strip())


def get_avg_pose_dict(target_list):
    """各personが期間内に属している時間の長いposeを返却する
    target_list(list): [{person_id: "pose1", ...}, {},...]
    """
    # person ごとのポーズカウント
    counts = defaultdict(Counter)  # {"person_id": {"pose1": 5, ...}}
    for pose_map in target_list:
        for person_id, pose in pose_map.items():
            counts[person_id][pose] += 1

    # personごとの属している時間がながいポーズ {"person_id": "pose1",...}
    representive_pose_map = {
        person_id: counter.most_common(1)[0][0] for person_id, counter in counts.items()
    }
    return representive_pose_map


def get_person_xydict(
    result, tracked_people_xys: dict[int, TrackedPerson]
) -> defaultdict:
    # result から中心点を取得する
    """person_id ごとの骨格点を取る"""
    person_xydict = defaultdict(dict)
    # 矩形のxy, パーツ別confidence, パーツ位置xy
    for boxes_xy, keypoints in zip(result.boxes.xyxy, result.keypoints):
        # for boxes_xy, conf, xy in zip(result.boxes.xyxy, result.keypoints.conf, result.keypoints.xy):
        # まず id を取得
        x1, y1, x2, y2 = map(int, boxes_xy[:4])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 中心座標
        # 最も近い id を取り出し、なければ skip
        person_id = _find_nearest_id(tracked_people_xys, cx, cy, threshold=100)
        if person_id is None:
            continue

        person_keypoints = PersonKeypoints.from_yolo_keypoints(keypoints)
        person_xydict[person_id] = person_keypoints
    return person_xydict


def _find_nearest_id(
    tracked_people_xys: dict[int, TrackedPerson], x: int, y: int, threshold: int
):
    nearest_id = None
    min_distance = float("inf")

    for id, tracked_person in tracked_people_xys.items():
        distance = math.sqrt(
            (tracked_person.center_x - x) ** 2 + (tracked_person.center_y - y) ** 2
        )
        if distance <= threshold and distance < min_distance:
            min_distance = distance
            nearest_id = id

    return nearest_id
