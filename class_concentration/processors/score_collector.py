from dataclasses import dataclass
from collections import defaultdict, Counter
from utils.utils import get_avg_dict, get_avg_pose_dict
from models.settings.settings import Settings


@dataclass
class FrameData:
    def __init__(self, frame_id, score_map, pose_map):
        self.frame_id = frame_id
        self.score_map = score_map  # {person_id: {pose_type: 0.2, ..}
        self.pose_map = pose_map  # {person_id: "pose1", ...}


class VideoScoreCollector:
    """1秒(fps分のframe数)のスコアの平均を取得後 window_size_secでさらに平均を取る
    prefix fps のリストには1秒毎のスコア,window_のリストにwindow_size_sec毎のスコアが入っている
    """

    def __init__(self, settings: Settings):
        self.data = []
        self.fps = settings.video_info.fps
        self.window_size_sec = (
            settings.window_size_sec
        )  # 全体平均スコアを取るウィンドウサイズ(秒)

        # 個人: fps単位の平均
        self.fps_personal_score_list = []  # fpsごとの person 単位の平均スコアを保持
        self.fps_personal_pose_list = []

        # 全体: window単位のすべてのユーザ平均スコア
        self.window_total_score_list = (
            []
        )  # [{"1": 0.2, "2": 0.4, ...., "total": 0.2}, .....]
        self.window_total_pose_list = []

    def add(self, frame_id, score_map, pose_map):
        # フレーム毎のデータをすべて insert していく
        self.data.append(FrameData(frame_id, score_map, pose_map))

        if frame_id % self.fps == 0:
            # fps ごとの平均値
            self.fps_personal_score_list.append(self._get_avg_score())
            # fps ごとのidごとの代表ポーズ
            self.fps_personal_pose_list.append(self._get_avg_pose())

    def calc_window_score(self):
        """window_size毎の各人の平均スコアおよび平均ポーズを取得する"""
        for i in range(0, len(self.fps_personal_score_list), self.window_size_sec):
            start_index = i
            if i + self.window_size_sec > len(self.fps_personal_score_list):
                end_index = len(self.fps_personal_score_list)
            else:
                end_index = i + self.window_size_sec

            avg_score_dict = get_avg_dict(
                self.fps_personal_score_list[start_index:end_index]
            )
            avg_pose_map = get_avg_pose_dict(
                self.fps_personal_pose_list[start_index:end_index]
            )

            self.window_total_score_list.append(avg_score_dict)
            self.window_total_pose_list.append(avg_pose_map)

    def _get_avg_score(self) -> dict:
        """
        fpsごとの平均値を返却
        target_list(list[FrameData]): [{person_id: {pose_type: 0.2, ..}...}, {},...]
        """
        # 各キーごとの合計値とカウントを保持する辞書
        sum_dict = defaultdict(float)
        count_dict = defaultdict(int)

        # 合計値とカウントを計算
        for framedata in self.data[-self.fps :]:
            score_map = framedata.score_map
            for key, value in score_map.items():
                sum_dict[key] += value
                count_dict[key] += 1

        # 平均値を計算
        avg_dict = {key: sum_dict[key] / count_dict[key] for key in sum_dict}
        return avg_dict

    def _get_avg_pose(self):
        """各personが期間内に属している時間の長いposeを返却する
        target_list(list): [{person_id: "pose1", ...}, {},...]
        """
        # person ごとのポーズカウント
        counts = defaultdict(Counter)  # {"person_id": {"pose1": 5, ...}}
        for framedata in self.data[-self.fps :]:
            pose_map = framedata.pose_map
            for person_id, pose in pose_map.items():
                counts[person_id][pose] += 1

        # personごとの属している時間がながいポーズ {"person_id": "pose1",...}
        representive_pose_map = {
            person_id: counter.most_common(1)[0][0]
            for person_id, counter in counts.items()
        }
        return representive_pose_map


class AudioScoreCollector:
    def __init__(self):
        self.window_total_audioscore_list = []
        self.speak_utterence_main_or_others_per_window = []
