from models.settings.settings import Settings
from moviepy.editor import VideoFileClip, AudioFileClip
from utils.utils import COLOR_MAP, SKELETON_NAME_PAIRS
import cv2
import os
from models.persons.tracked_person import TrackedPerson
import time


class OutputVideoProcessor:
    def __init__(
        self,
        settings: Settings,
        output_file_name: str = "final_video.mp4",
    ):
        self.settings = settings
        # 中間ファイル
        self.tmp_video_path = os.path.join(self.settings.base_path, "output.mp4")
        self.tmp_audio_path = os.path.join(self.settings.base_path, "output.wav")
          
        self.output_path = os.path.join(self.settings.base_path, output_file_name)

        self.width = self.settings.video_info.width
        self.height = self.settings.video_info.height

    def _save(self):
        video_clip = VideoFileClip(self.tmp_video_path)
        # 音声ファイルを読み込む
        audio_clip = AudioFileClip(self.tmp_audio_path)  # ここで音声ファイルを指定
        # ビデオクリップに音声をセット
        video_clip = video_clip.set_audio(audio_clip)
        # 音声付きのビデオファイルとして保存
        video_clip.write_videofile(self.output_path, codec="libx264", audio_codec="aac")

        # 中間ファイルの削除
        if os.path.exists(self.tmp_video_path):
            os.remove(self.tmp_video_path)
        if os.path.exists(self.tmp_audio_path):
            os.remove(self.tmp_audio_path)

    def draw(
        self,
        tracked_people_xys: dict[int, TrackedPerson],
        tracked_person_keypoints_list: list,
        video_score_list: list,
        audio_score_list: list,
    ):
        cap = cv2.VideoCapture(self.settings.video_info.video_path)
        out = cv2.VideoWriter(
            os.path.join(
                self.settings.base_path, self.tmp_video_path
            ),  # 音声を重ねる前
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.settings.video_info.fps,
            (self.settings.video_info.width, self.settings.video_info.height),
        )

        frame_cnt = 0
        window_index = 0  # 何番目の window か
        window_frame_size = self.settings.video_info.fps * self.settings.window_size_sec

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("動画の終端に到達したため終了")
                    break

                # 毎フレーム書き換える
                tracked_person_keypoints = tracked_person_keypoints_list[frame_cnt]

                # window 毎に書き換える
                video_score_map: dict = video_score_list[window_index]
                audio_score: float = audio_score_list[window_index]  # total のみ、list

                # 骨格
                for (
                    person_id,
                    person_kp,
                ) in tracked_person_keypoints.items():  # id: PersonKeypoints

                    all_keypoints = (
                        person_kp.get_all_keypoints()
                    )  # {"nose": [x, y], ...}
                    for keypoint in all_keypoints.values():
                        cv2.circle(
                            frame,
                            (int(keypoint[0]), int(keypoint[1])),
                            4,
                            COLOR_MAP["orange"],
                            thickness=4,
                            lineType=cv2.LINE_AA,
                        )
                    for skeleton_pair in SKELETON_NAME_PAIRS:
                        if not (
                            skeleton_pair[0] in all_keypoints.keys()
                            and skeleton_pair[1] in all_keypoints.keys()
                        ):
                            continue
                        if len(all_keypoints[skeleton_pair[0]]) > 2:
                            print(all_keypoints[skeleton_pair[0]])
                        x1, y1 = all_keypoints[skeleton_pair[0]]
                        x2, y2 = all_keypoints[skeleton_pair[1]]
                        cv2.line(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            COLOR_MAP["orange"],
                            1,
                        )

                ######## person_id 単位の集中度
                for person_id, video_score in video_score_map.items():
                    # 描画位置
                    if person_id != "total":
                        tp: TrackedPerson = tracked_people_xys[int(person_id)]
                        x = tp.center_x
                        y = tp.center_y
                        person_score = round(video_score, 2)
                        score_text = f"{person_id}: {person_score}"

                        cv2.rectangle(
                            frame,
                            (int(x), int(y - 20)),
                            (int(x + 70), int(y + 20)),
                            COLOR_MAP["black"],
                            -1,
                        )
                        # font color
                        if person_score < self.settings.posing_score_threhold:
                            font_color = COLOR_MAP["red"]
                        else:
                            font_color = COLOR_MAP["green"]
                        cv2.putText(
                            frame,
                            score_text,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            font_color,
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            score_text,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            font_color,
                            2,
                            cv2.LINE_AA,
                        )

                #### 全体集中度の描画
                # video
                # font color
                total_score = round(video_score_map["total"], 2)
                if total_score < self.settings.posing_score_threhold:
                    font_color = COLOR_MAP["red"]
                else:
                    font_color = COLOR_MAP["green"]
                total_video_score_text = "total score from video: "
                cv2.rectangle(
                    frame,
                    (int(self.width * 0.5 / 10), int(self.height * 6 / 10)),
                    (int(self.width * 3.5 / 10), int(self.height * 7 / 10)),
                    COLOR_MAP["black"],
                    -1,
                )
                cv2.putText(
                    frame,
                    total_video_score_text,
                    (int(self.width * 0.5 / 10), int(self.height * 6.5 / 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    COLOR_MAP["white"],
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"{total_score}",
                    (int(self.width * 2.5 / 10), int(self.height * 6.5 / 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    font_color,
                    2,
                    cv2.LINE_AA,
                )

                # audio
                # font color
                audio_score = round(audio_score, 2)
                if audio_score < self.settings.posing_score_threhold:
                    font_color = COLOR_MAP["red"]
                else:
                    font_color = COLOR_MAP["green"]
                total_audio_score_text = "total score from audio: "
                cv2.rectangle(
                    frame,
                    (int(self.width * 0.5 / 10), int(self.height * 7 / 10)),
                    (int(self.width * 3.5 / 10), int(self.height * 8 / 10)),
                    COLOR_MAP["black"],
                    -1,
                )
                cv2.putText(
                    frame,
                    total_audio_score_text,
                    (int(self.width * 0.5 / 10), int(self.height * 7.5 / 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    COLOR_MAP["white"],
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"{audio_score}",
                    (int(self.width * 2.5 / 10), int(self.height * 7.5 / 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    font_color,
                    2,
                    cv2.LINE_AA,
                )
                out.write(frame)
                # 値の更新
                if frame_cnt != 0 and frame_cnt % window_frame_size == 0:
                    window_index += 1
                frame_cnt += 1
        except KeyboardInterrupt:
            print("\n動画の保存を終了します...")

        finally:
            cap.release()
            out.release()  # ここで動画保存
            time.sleep(5)  # 書き込み反映のために少し待つ
            # moviepy を使って動画と音声を重ねる
            self._save()
