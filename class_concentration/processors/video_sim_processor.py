from models.persons.ideal_pose import IdealPose
from models.settings.settings import Settings
from processors.score_collector import VideoScoreCollector
from collections import defaultdict
from models.persons.relative_xy import RelativeXY
import numpy as np
from scipy.spatial import distance
from models.persons.tracked_person import TrackedPerson
import cv2
from utils.utils import show_detect_persons, get_person_xydict


class VideoSimilarityProcessor:
    def __init__(
        self,
        model,  # YOLO model
        ideal_pose_list: list[IdealPose],
        video_score_collector: VideoScoreCollector,
        settings: Settings,
    ):

        self.model = model
        self.ideal_pose_list = ideal_pose_list

        self.settings = settings
        self.sigma = self.settings.sigma
        self.pose_similar_threshold = (
            self.settings.posing_score_threhold
        )  # これを下回る場合 "other" ポーズを取っていたことになる

        # video info
        self.window_size_sec = settings.window_size_sec

        # trackedPersonの中心座標が入っている
        self.tracked_people_xys: dict[int, TrackedPerson] = {}
        self.prev_tracked_people_score_map = defaultdict(dict)
        # trackedPersonの骨格点のビデオフレーム長のリスト。最後の描写に利用
        self.tracked_person_keypoints_list: list = []

        self.video_score_collector = video_score_collector

    # ideal_pose_list: list[IdealPose] # 理想のポーズの座標、  pose_name, relative_xy, cov_i

    def _get_similarity(
        self, frame, tracked_person_keypoints: defaultdict
    ) -> defaultdict:
        """
        集中度を計算する person_id, pose ごとの類似度を持つ dict を返す
        Args:
          frame: フレーム
          tracked_person_keypoints_list(list): personごとの骨格点map [{person_id: Keypoints, ....}]
        Returns(dict):
          {person_id: {pose_name: 0.2, ...},...}
        """
        if len(self.ideal_pose_list) == 0 or self.ideal_pose_list == None:
            print("error")
            return AssertionError

        relative_xy_map = {}  # 鼻との相対位置の xys が person_id 別に入っている

        for person_id, keypoints in tracked_person_keypoints.items():
            # person 別に relative_xys を格納
            relative_xy_map[person_id] = RelativeXY.from_keypoints(keypoints)

        # {person_id: {pose1: 0.7, pose2: }
        return self._get_person_id2similarities(relative_xy_map)

    def _get_person_id2similarities(
        self, relative_xy_map: dict[int, RelativeXY]
    ) -> defaultdict:
        pose2similarity = {}
        for ideal_pose in self.ideal_pose_list:
            id2similarity = self._calc_similarity(
                relative_xy_map, ideal_pose
            )  # , sigma=9) # {id: {pose_name: score}, ...}
            pose2similarity[ideal_pose.pose_name] = (
                id2similarity  # person_id ごとのスコア
            )

        # 各person_idにポーズごとの類似度を格納していく
        similarities = defaultdict(dict)
        # person_id ごとに各ポーズのスコアを入れる
        for pose_name, similarity in pose2similarity.items():
            for person_id, score in similarity.items():
                similarities[person_id][pose_name] = score

        return similarities

    def _calc_similarity(
        self, relative_xy_map: dict[int, RelativeXY], ideal_pose: IdealPose
    ):
        """
        - 現在は mahalanobis 距離で計算している。将来的に変更したほうがいいかも
        """
        ideal_xy_flatten = ideal_pose.relative_xy.flatten_relative_xys()
        distance_map = {}
        for person_id, relative_xy in relative_xy_map.items():
            xy_flatten = relative_xy.flatten_relative_xys()
            d = distance.mahalanobis(ideal_xy_flatten, xy_flatten, ideal_pose.cov_i)
            distance_map[person_id] = np.exp(-0.5 * (d / self.sigma) ** 2)
        return distance_map

    def _get_max_similarity_and_pose(self, similarities: defaultdict):
        """各person_id ごとの最大類似度をその地点のスコアとする
        similarity_map(dict): {person_id: {pose_name: 0.2, ...},...}
        """
        self.score_map = {}  # 各 frame の各ユーザの score
        self.pose_map = defaultdict(int)
        for person_id, pose_similarity_map in similarities.items():
            # for debug
            # if person_id == 10:
            #   print(pose_similarity_map)
            pose = "other"
            score = max(pose_similarity_map.values())
            self.score_map[person_id] = score
            if score >= self.pose_similar_threshold:
                pose = max(pose_similarity_map, key=pose_similarity_map.get)

            self.pose_map[person_id] = pose

    def _save_frame_score(self, frame_cnt, similarities: defaultdict):
        """各フレームにおけるスコアとポーズリストを保存する"""
        self._get_max_similarity_and_pose(similarities)
        self.video_score_collector.add(
            frame_id=frame_cnt, score_map=self.score_map, pose_map=self.pose_map
        )

    def _get_user_position(self, results) -> dict[int, TrackedPerson]:
        """
        user position から id を採番する
        基準位置はユーザの矩形の中心点を取る
        Arg:
          results: YOLO での解析結果
          detect_xy(dict): 検出対象位置、パーツを持つ dict
          tracked_people_xys(dict): ユーザの現在の基準位置。描画矩形の中心位置を取得している。 {0: [x, y], 1: [x, y], ...}
        Return:
          tracked_people_xys(dict): {0: [x, y], 1: [x, y], ...}
        """
        detect_xy = self.settings.detect_area
        tracked_people = self.tracked_people_xys
        detection_people_xys = []
        for result in results:
            keypoints = result.keypoints.xy  # (人数, 関節数, 2)
            for i, box in enumerate(result.boxes.xyxy):
                kpts = keypoints[i].cpu().numpy()

                # 関節が5点以上検出されているかどうかチェック
                valid = (
                    ~np.isnan(kpts).any(axis=1) & (kpts[:, 0] > 0) & (kpts[:, 1] > 0)
                ).sum()
                if valid < 5:
                    continue  # 検出点が少ないならスキップ

                x1, y1, x2, y2 = map(int, box[:4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 中心座標
                # 検出対象位置に cx, cy が存在しない場合はやめる
                if not (
                    detect_xy.min[0] <= cx
                    and cx <= detect_xy.max[0]
                    and detect_xy.min[1] <= cy
                    and cy <= detect_xy.max[1]
                ):
                    continue
                detection_people_xys.append([cx, cy])

        for cx, cy in detection_people_xys:
            # すでに登録している場合、場所を少し補正する
            matched_id = None
            for pid, tracked_person in tracked_people.items():
                if (
                    np.linalg.norm(
                        np.array([cx, cy])
                        - np.array([tracked_person.center_x, tracked_person.center_y])
                    )
                    < self.settings.threshold_dist
                ):
                    matched_id = pid
                    new_cx, new_cy = (cx + tracked_person.center_x) // 2, (
                        cy + tracked_person.center_y
                    ) // 2  # 中心座標
                    tracked_people[pid].center_x = new_cx
                    tracked_people[pid].center_y = new_cy
                    break

            # 新規登録
            if matched_id is None:
                next_id = len(tracked_people.keys()) + 1
                tracked_person = TrackedPerson(id=next_id, center_x=cx, center_y=cy)
                tracked_people[next_id] = tracked_person

        return tracked_people

    def run(self):
        # はじめの N 秒間でユーザの基準位置をとる
        id_check_frame_size = (
            self.settings.video_info.fps * self.settings.id_fetch_duration_sec
        )
        id_fetch_frame_interval = (
            self.settings.video_info.fps * self.settings.id_fetch_interval
        )
        frame_cnt = 0

        cap = cv2.VideoCapture(self.settings.video_info.video_path)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("動画の終端に到達したため終了")
                    break

                results = self.model(frame, verbose=False)

                # はじめだけ入る
                if (frame_cnt <= id_check_frame_size) and (
                    frame_cnt % id_fetch_frame_interval == 0
                ):
                    self.tracked_people_xys = self._get_user_position(
                        results=results
                    )  # {0:(x, y), 1: (x,y),....}

                    filtered_people_xys = {
                        pid: tracked_person
                        for pid, tracked_person in self.tracked_people_xys.items()
                        if not tracked_person.is_mask_area(self.settings.mask_area)
                    }

                # track people を確認
                if frame_cnt == id_check_frame_size:
                    print(
                        len(self.tracked_people_xys.keys()),
                        len(filtered_people_xys.keys()),
                        filtered_people_xys,
                    )
                    # 一回表示する、ダメだったら  Ctrl + C 2 回おす
                    tmp_frame = frame.copy()
                    show_detect_persons(tmp_frame, filtered_people_xys)
                    del tmp_frame

                # track する人の id とスコア 前のフレームのを引き継ぐ
                tracked_people_score_map = self.prev_tracked_people_score_map.copy()
                # person_id ごとの骨格点を抽出、あとで描画するためこの時点では relativeXY ではなくkeypoint
                tracked_person_keypoints = get_person_xydict(
                    results[0], self.tracked_people_xys
                )
                tracked_person_keypoints.update(tracked_person_keypoints)
                self.tracked_person_keypoints_list.append(tracked_person_keypoints)

                ##### video similarity #######################################################################
                # person_id ごとに、各ポーズとの類似度がはいっている、get similarity内部でrelativeXYに変換する
                # defaultdict(<class 'dict'>, {1: {'look_forward': np.float64(0.02202171), 'writing_note': np.float64(0.0013893)},...})
                current_similarities = self._get_similarity(
                    frame, tracked_person_keypoints
                )
                tracked_people_score_map.update(current_similarities)
                # 計算できていない person がいる場合、1つ前のフレームの score を引き継ぎたいため
                self.prev_tracked_people_score_map = tracked_people_score_map
                self._save_frame_score(frame_cnt, tracked_people_score_map)

                frame_cnt += 1

        except KeyboardInterrupt:
            print("\n動画の保存を終了します...")

        finally:
            cap.release()

        self.video_score_collector.calc_window_score()  # window_total_score_list / window_total_pose_list が存在

    def extract_tracked_person_info(self):
        """描画に必要な tracked_person の骨格点、中心点を返却する"""
        return self.tracked_people_xys, self.tracked_person_keypoints_list
