## pip install ultralytics -U --force-reinstall
## from: https://github.com/bharath5673/StrongSORT-YOLO
##
## Enhanced with class-based tracking and trajectory visualization
## Features:
## - Class-specific object detection and tracking
## - Class-based heatmap generation
## - Class-based trajectory visualization
## - Configurable target classes
##
## Usage examples:
## Standard models:
## python yolo_multi_model_refactored.py --source video.mp4 --track --classes "0,2" --tracker reid
## python yolo_multi_model_refactored.py --source 0 --track --count --classes "0,1,2,5" --tracker reid
##
## Tracker options:
## --tracker reid   : Use ReIDTracker (botsort + bytetrack) - Recommended
## --tracker yolo   : Use YOLO built-in tracker
##
## Custom models (Adult/Child classification):
## python yolo_multi_model_refactored.py --source video.mp4 --track --model "path/to/custom_model.pt" --dataset-yaml "adult_child.yaml" --classes "0,1" --tracker reid
## python yolo_multi_model_refactored.py --source 0 --track --count --model "best.pt" --dataset-yaml "adult_child.yaml" --tracker reid
##
## Output files for Adult/Child model:
## - video_heatmap_Child_class0.jpg (Child class heatmap)
## - video_heatmap_Adult_class1.jpg (Adult class heatmap)
## - video_trajectory_Child_class0.jpg (Child class trajectory)
## - video_trajectory_Adult_class1.jpg (Adult class trajectory)
## - video_heatmap_all_humans.jpg (Combined Child+Adult heatmap)
## - video_trajectory_all_humans.jpg (Combined Child+Adult trajectory)

import os
import cv2
import numpy as np
import time
from collections import Counter
import pandas as pd
import argparse
from multiprocessing import Pool
from yolo_pipeline_processor import YOLOPipelineProcessor


def process_video(args):
    """動画を処理するメイン関数"""
    print(args)
    source = args["source"]
    track_ = args["track"]
    count_ = args["count"]

    config_path = args.get("config", "config.yaml")

    # 対象クラスの設定
    target_classes = args.get("classes", [0, 2])  # デフォルトは person=0, car=2
    if isinstance(target_classes, str):
        # 文字列の場合はカンマ区切りで分割
        target_classes = [int(x.strip()) for x in target_classes.split(",")]

    print(f"Target classes: {target_classes}")

    # カスタムモデルとデータセットYAMLの処理
    model_path = args.get("model")
    dataset_yaml = args.get("dataset_yaml")
    tracker_type = args.get("tracker", "reid")

    print(f"Using tracker: {tracker_type}")

    # YOLOパイプラインプロセッサーの初期化
    if dataset_yaml and model_path:
        # カスタムデータセットで学習したモデルを使用
        print(f"Using custom model: {model_path}")
        print(f"Using dataset YAML: {dataset_yaml}")
        processor = YOLOPipelineProcessor.create_with_custom_dataset(
            model_path=model_path,
            dataset_yaml_path=dataset_yaml,
            config_path=config_path,
            target_classes=target_classes,
        )
    elif model_path:
        # カスタムモデルのみ指定（標準クラス使用）
        print(f"Using custom model with standard classes: {model_path}")
        processor = YOLOPipelineProcessor(
            config_path=config_path,
            target_classes=target_classes,
            model_path=model_path,
        )
    else:
        # 標準設定
        processor = YOLOPipelineProcessor(config_path, target_classes)

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

    # 動画名に基づいた出力ディレクトリを作成
    video_output_dir = f"output/{input_video_name}"
    if processor.config.get("output.create_output_dir", True):
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

    out = cv2.VideoWriter(
        f"{video_output_dir}/{input_video_name}_output.mp4",
        fourcc,
        output_fps,
        (frame_width, frame_height),
    )
    labels_file_path = os.path.abspath(
        f"./{video_output_dir}/{input_video_name}_labels.txt"
    )

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

        # 1フレーム目を保存（ヒートマップ用）
        if frameId == 1 and track_:
            processor.set_first_frame(frame1)

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
                for key, val in processor.class_names.items():
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

    # クラス別ヒートマップと軌跡画像の生成（トラッキングが有効な場合のみ）
    if track_:
        try:
            generated_files = []

            # クラス別ヒートマップの生成
            if processor.config.get("heatmap.generate_class_heatmaps", True):
                class_heatmaps = processor.generate_heatmaps(video_output_dir)
                for class_id, heatmap in class_heatmaps.items():
                    class_name = processor.class_names.get(
                        class_id, f"class_{class_id}"
                    )
                    file_path = f"{video_output_dir}/{input_video_name}_heatmap_{class_name}_class{class_id}.jpg"
                    generated_files.append(file_path)

            # クラス別軌跡ヒートマップの生成
            if processor.config.get("heatmap.generate_trajectory_heatmaps", True):
                trajectory_heatmaps = processor.generate_trajectory_heatmaps(
                    video_output_dir
                )
                for class_id, trajectory_heatmap in trajectory_heatmaps.items():
                    class_name = processor.class_names.get(
                        class_id, f"class_{class_id}"
                    )
                    file_path = f"{video_output_dir}/{input_video_name}_trajectory_{class_name}_class{class_id}.jpg"
                    generated_files.append(file_path)

            # 全人間統合ヒートマップの生成
            if processor.config.get("heatmap.generate_combined_heatmaps", True):
                # カスタムクラスの場合は全クラスを統合、標準の場合はperson(0)のみ
                if hasattr(processor, "custom_classes") and processor.custom_classes:
                    # カスタムクラス（Child + Adult）の統合
                    combined_heatmaps = processor.generate_combined_heatmaps(
                        video_output_dir,
                        combine_classes=processor.get_target_classes(),
                        label="all_humans",
                    )
                    generated_files.append(
                        f"{video_output_dir}/{input_video_name}_heatmap_all_humans.jpg"
                    )
                    generated_files.append(
                        f"{video_output_dir}/{input_video_name}_trajectory_all_humans.jpg"
                    )
                else:
                    # 標準クラスの場合はperson(0)のみ
                    person_classes = [0]  # person class
                    if 0 in processor.get_target_classes():
                        combined_heatmaps = processor.generate_combined_heatmaps(
                            video_output_dir,
                            combine_classes=person_classes,
                            label="all_persons",
                        )
                        generated_files.append(
                            f"{video_output_dir}/{input_video_name}_heatmap_all_persons.jpg"
                        )
                        generated_files.append(
                            f"{video_output_dir}/{input_video_name}_trajectory_all_persons.jpg"
                        )

            # 統計情報の保存と出力
            processor.save_statistics(video_output_dir)
            stats = processor.get_statistics()
            print(f"\n=== Processing Statistics ===")
            print(f"Total frames processed: {stats['total_frames']}")
            print(f"Target classes: {stats['target_classes']}")
            print(f"Class names: {stats['class_names']}")
            print(f"Detection stats: {stats['detection_stats']}")
            print(f"Heatmap stats: {stats['heatmap_stats']}")

            # 統計ファイルも生成ファイルリストに追加
            if processor.config.get("output.statistics.save_json", True):
                generated_files.append(f"{video_output_dir}/statistics.json")
            if processor.config.get("output.statistics.save_csv", True):
                generated_files.append(f"{video_output_dir}/detection_stats.csv")
                generated_files.append(f"{video_output_dir}/heatmap_stats.csv")

            if generated_files:
                print(f"\n=== Generated Files ===")
                for file_path in generated_files:
                    if os.path.exists(file_path):
                        print(f"  ✓ {file_path}")
                    else:
                        print(f"  ✗ {file_path} (not found)")

        except Exception as e:
            print(f"Error generating class-based heatmaps: {e}")
            import traceback

            traceback.print_exc()

    print(f"Processing completed. All outputs saved to: {video_output_dir}/")
    print(f"Main video: {video_output_dir}/{input_video_name}_output.mp4")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="YOLO Multi-Model Processing with Class-based Tracking"
    )
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
    parser.add_argument(
        "--classes",
        type=str,
        default="0,2",
        help="Target class IDs (comma-separated, e.g., '0,2' for person and car)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Custom model path (overrides config file)",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        help="Dataset YAML file path for custom classes (e.g., adult_child.yaml)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        choices=["reid", "yolo"],
        default="reid",
        help="Tracking method: 'reid' for ReIDTracker (botsort+bytetrack), 'yolo' for YOLO built-in tracker",
    )

    args = parser.parse_args()

    # 引数を辞書形式に変換
    args_dict = {
        "source": args.source,
        "track": args.track,
        "count": args.count,
        "config": args.config,
        "classes": args.classes,
        "model": args.model,
        "dataset_yaml": args.dataset_yaml,
        "tracker": args.tracker,
    }

    process_video(args_dict)


if __name__ == "__main__":
    main()
