## pip install ultralytics -U --force-reinstall
## from: https://github.com/bharath5673/StrongSORT-YOLO

import os
import cv2
import numpy as np
import time
from collections import Counter
import pandas as pd
import argparse
from multiprocessing import Pool
from yolo_processor import YOLOProcessor


def process_video(args):
    """動画を処理するメイン関数"""
    print(args)
    source = args["source"]
    track_ = args["track"]
    count_ = args["count"]

    config_path = args.get("config", "config.yaml")

    # YOLOプロセッサーの初期化
    processor = YOLOProcessor(config_path)

    # 動画キャプチャの設定
    cap = cv2.VideoCapture(int(source) if source == "0" else source)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # プロセッサーにフレームサイズを設定
    processor.set_frame_size(frame_width, frame_height)

    # 出力設定（設定ファイルから取得）
    output_fps = processor.config.get("output.fps", 15)
    codec = processor.config.get("output.codec", "mp4v")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    input_video_name = os.path.splitext(os.path.basename(source))[0]

    if processor.config.get("output.create_output_dir", True):
        if not os.path.exists("output"):
            os.makedirs("output")

    out = cv2.VideoWriter(
        f"output/{input_video_name}_output.mp4",
        fourcc,
        output_fps,
        (frame_width, frame_height),
    )
    labels_file_path = os.path.abspath(f"./output/{input_video_name}_labels.txt")

    if not cap.isOpened():
        print(f"Error: Could not open video file {source}.")
        return

    frameId = 0
    start_time = time.time()
    fps_str = str()

    while True:
        frameId += 1
        ret, frame = cap.read()
        if not ret:
            break

        frame1 = frame.copy()

        # フレームを処理
        processed_frame, bboxes = processor.process_frame(
            frame1, track=track_, use_heatmap=track_
        )

        # ラベルファイルに保存
        if track_:
            processor.save_labels(bboxes, frameId, labels_file_path)

        # カウント機能
        if not track_ and count_:
            print(
                "[INFO] count works only when objects are tracking.. so use: --track --count"
            )
            break

        if track_ and count_:
            itemDict = {}
            try:
                df = pd.read_csv(labels_file_path, header=None, sep=r"\s+")
                df = df.iloc[:, 0:3]
                df.columns = ["frameid", "class", "trackid"]
                df = df[["class", "trackid"]]
                df = (
                    df.groupby("trackid")["class"]
                    .apply(list)
                    .apply(lambda x: sorted(x))
                ).reset_index()
                df["class"] = df["class"].apply(
                    lambda x: Counter(x).most_common(1)[0][0]
                )
                vc = df["class"].value_counts()
                vc = dict(vc)

                vc2 = {}
                for key, val in enumerate(processor.names):
                    vc2[key] = val
                itemDict = dict((vc2[key], value) for (key, value) in vc.items())
                itemDict = dict(sorted(itemDict.items(), key=lambda item: item[0]))
            except:
                pass

            # オーバーレイ表示
            display = processed_frame.copy()
            h, w = processed_frame.shape[0], processed_frame.shape[1]
            x1, y1, x2, y2 = 10, 10, 10, 70
            txt_size = cv2.getTextSize(str(itemDict), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[
                0
            ]
            cv2.rectangle(
                processed_frame, (x1, y1 + 1), (txt_size[0] * 2, y2), (0, 0, 0), -1
            )
            cv2.putText(
                processed_frame,
                "{}".format(itemDict),
                (x1 + 10, y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (210, 210, 210),
                2,
            )
            cv2.addWeighted(processed_frame, 0.7, display, 1 - 0.7, 0, processed_frame)

        # FPS計算と表示
        fps_update_interval = processor.config.get("processing.fps_update_interval", 10)
        if frameId % fps_update_interval == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps_current = fps_update_interval / elapsed_time
            fps_str = f"FPS: {fps_current:.2f}"
            start_time = time.time()

        cv2.putText(
            processed_frame,
            fps_str,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        # 動画に書き込み
        out.write(processed_frame)

    # リソースの解放
    cap.release()
    out.release()
    print(
        f"Processing completed. Output saved to: output/{input_video_name}_output.mp4"
    )


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="YOLO Multi-Model Processing")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (file path or camera index)",
    )
    parser.add_argument("--track", action="store_true", help="Enable object tracking")
    parser.add_argument("--count", action="store_true", help="Enable object counting")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Configuration file path"
    )

    args = parser.parse_args()

    # 引数を辞書形式に変換
    args_dict = {
        "source": args.source,
        "track": args.track,
        "count": args.count,
        "config": args.config,
    }

    process_video(args_dict)


if __name__ == "__main__":
    main()
