import os
import pandas as pd
from processors.score_collector import AudioScoreCollector, VideoScoreCollector
from collections import defaultdict
import numpy as np


class ScoreProcessor:
    def __init__(
        self,
        video_score_collector: VideoScoreCollector,
        audio_score_collector: AudioScoreCollector,
        output_path_dir: str,
    ):
        self.window_total_pose_list = video_score_collector.window_total_pose_list
        self.window_total_score_list = video_score_collector.window_total_score_list

        self.window_total_audioscore_list = (
            audio_score_collector.window_total_audioscore_list
        )
        self.speak_utterence_main_or_others_per_window = (
            audio_score_collector.speak_utterence_main_or_others_per_window
        )

        self.output_path_dir = output_path_dir

        self.pose_count_list = (
            []
        )  # window_size毎のそのポーズに属した人数が入っている。csv出力する

    def calculate(self):
        # window_size毎のポーズをカウント
        for pose_map in self.window_total_pose_list:
            pose_count = defaultdict(int)
            for pose in pose_map.values():
                pose_count[pose] += 1

            self.pose_count_list.append(pose_count)

        # window_size毎の全体の平均スコアを取る
        for avg_score_dict in self.window_total_score_list:
            total_score = np.array(list(avg_score_dict.values())).mean()
            # 集中してる人数 / 全体の人数 (optional)
            # total_score = calc_total_concenterate(avg_score_dict, threshold = threshold)
            avg_score_dict["total"] = total_score

    def write_csv(self):
        # output_path
        videoscore_path = os.path.join(self.output_path_dir, "video_score.csv")
        pose_path = os.path.join(self.output_path_dir, "pose_count.csv")
        audioscore_path = os.path.join(self.output_path_dir, "audio_score.csv")

        # video score
        pd.DataFrame(self.window_total_score_list).to_csv(videoscore_path, index=False)

        # video pose
        pose_count_df = pd.DataFrame(self.pose_count_list)
        pose_count_df = pose_count_df.fillna(0)
        pose_count_df = pose_count_df.astype(int)
        pose_count_df.to_csv(pose_path, index=False)

        # audio score
        audio_df = pd.DataFrame(self.speak_utterence_main_or_others_per_window)
        audio_df["score"] = self.window_total_audioscore_list
        # 欠損値を埋める
        audio_df = audio_df.fillna(0.0)
        audio_df.to_csv(audioscore_path, index=False)

    def get_scores(self):
        return self.window_total_score_list, self.window_total_audioscore_list
