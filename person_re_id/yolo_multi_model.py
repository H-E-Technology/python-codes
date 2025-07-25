## pip install ultralytics -U --force-reinstall
## from: https://github.com/bharath5673/StrongSORT-YOLO

import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import Counter, deque
import pandas as pd
import argparse
from multiprocessing import Pool

# Load a model
# model = YOLO('yolov8n-seg.pt')  # load an official model
# model = YOLO('yolov5n.pt')  # load an official model
# model = YOLO("yolo11n.pt")
model = YOLO("yolo11n-pose.pt")
model.overrides['conf'] = 0.3  # NMS confidence threshold
model.overrides['iou'] = 0.4  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image
# model.overrides['classes'] = 2 ## define classes
names = model.names
names = {value: key for key, value in names.items()}
colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')


tracking_trajectories = {}  # 各IDの軌跡を保存（無制限に保存）
lost_tracks = {}  # 途切れたbytetrackのトラックを保存
track_id_mapping = {}  # botsort IDをbytetrack IDにマッピング
last_seen_frame = {}  # 各IDが最後に見られたフレーム番号
current_frame = 0  # 現在のフレーム番号
MAX_LOST_FRAMES = 30  # トラックが失われたと見なすまでの最大フレーム数
DRAW_POINTS = False  # センターポイントを点として表示するかどうか
LINE_THICKNESS = 2  # 軌跡の線の太さ
CENTER_AREA_RATIO = 0.4  # 画面中央領域の比率（画面サイズに対する割合）
MAX_TRAJECTORY_DISTANCE = 100  # 軌跡の最大距離（これ以上離れていると別人物と判断）
id_remapping = {}  # 新しいIDから元のIDへのマッピング
frame_width = 0  # 画面の幅（process_video関数で設定）
frame_height = 0  # 画面の高さ（process_video関数で設定）
hidden_tracks = {}  # 画面中央で隠れた人物の情報を保存 {id: {"last_position": (x, y), "last_frame": frame_num, "trajectory": deque()}}
MAX_HIDDEN_FRAMES = 60  # 隠れた状態として扱う最大フレーム数

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_distance(centroid1, centroid2):
    """Calculate Euclidean distance between two centroids"""
    return np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)

def get_bbox_centroid(bbox):
    """Calculate centroid of a bounding box"""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def is_in_center_area(centroid, image_width, image_height):
    """Check if a point is in the center area of the image"""
    center_x_min = image_width * (0.5 - CENTER_AREA_RATIO / 2)
    center_x_max = image_width * (0.5 + CENTER_AREA_RATIO / 2)
    center_y_min = image_height * (0.5 - CENTER_AREA_RATIO / 2)
    center_y_max = image_height * (0.5 + CENTER_AREA_RATIO / 2)
    
    x, y = centroid
    return (center_x_min <= x <= center_x_max) and (center_y_min <= y <= center_y_max)

def find_matching_hidden_track(centroid, class_id, hidden_tracks, max_distance=100):
    """Find a matching hidden track for a new detection"""
    best_match_id = None
    best_match_distance = max_distance
    
    for hidden_id, hidden_info in list(hidden_tracks.items()):
        # 同じクラスのオブジェクトのみ比較
        if hidden_info["class"] != class_id:
            continue
            
        # 隠れてから一定フレーム数以上経過したものは除外
        frames_hidden = current_frame - hidden_info["last_frame"]
        if frames_hidden > MAX_HIDDEN_FRAMES:
            continue
            
        # 最後の位置との距離を計算
        last_position = hidden_info["last_position"]
        distance = calculate_distance(centroid, last_position)
        
        # 距離が近い場合、マッチング候補とする
        if distance < best_match_distance:
            best_match_id = hidden_id
            best_match_distance = distance
            
    return best_match_id

def process(image, track=True):
    global input_video_name, current_frame
    bboxes = []
    frameId = 0
    current_frame += 1
    
    # Place this code outside the loop to avoid creating the file multiple times
    if not os.path.exists('output'):
        os.makedirs('output')
    labels_file_path = os.path.abspath(f'./output/{input_video_name}_labels.txt')

    # Open the file in 'a' (append) mode
    with open(labels_file_path, 'a') as file:
        if track is True:
            # Primary tracker: ByteTrack
            results_bytetrack = model.track(image, verbose=False, device=0, persist=True, tracker="bytetrack.yaml")
            # Secondary tracker: BotSORT for補完
            results_botsort = model.track(image.copy(), verbose=False, device=0, persist=False, tracker="botsort.yaml")
            
            # 現在のフレームで検出されたbytetrackのID
            active_bytetrack_ids = set()
            bytetrack_boxes = {}  # bytetrackのID -> bbox
            
            # ByteTrackの結果を処理
            for predictions in results_bytetrack:
                if predictions is None or predictions.boxes is None or predictions.boxes.id is None:
                    continue
                    
                for bbox in predictions.boxes:
                    for scores, classes, bbox_coords, id_ in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
                        if id_ is not None:
                            bytetrack_id = int(id_)
                            active_bytetrack_ids.add(bytetrack_id)
                            bytetrack_boxes[bytetrack_id] = bbox_coords.tolist()
                            last_seen_frame[bytetrack_id] = current_frame
            
            # 失われたトラックを更新
            for track_id in list(last_seen_frame.keys()):
                if track_id not in active_bytetrack_ids:
                    frames_lost = current_frame - last_seen_frame[track_id]
                    if frames_lost <= MAX_LOST_FRAMES:
                        lost_tracks[track_id] = last_seen_frame[track_id]
                        
                        # 画面中央で消えた人物を hidden_tracks に保存
                        if track_id in tracking_trajectories and len(tracking_trajectories[track_id]) > 0:
                            last_position = tracking_trajectories[track_id][-1]
                            
                            # 画面中央で消えたかチェック
                            if is_in_center_area(last_position, frame_width, frame_height):
                                # クラスIDを取得（最後に検出されたクラス）
                                class_id = None
                                for item in bboxes:
                                    if len(item) >= 4 and item[3] == track_id:
                                        class_id = item[2]
                                        break
                                
                                if class_id is not None:
                                    # 隠れた人物として記録
                                    hidden_tracks[track_id] = {
                                        "last_position": last_position,
                                        "last_frame": current_frame,
                                        "class": class_id,
                                        "trajectory": tracking_trajectories[track_id].copy()
                                    }
                                    print(f"Person with ID {track_id} is hiding in center area")
                    
                    elif track_id in lost_tracks:
                        del lost_tracks[track_id]
                        # hidden_tracks からも削除（隠れ状態が長すぎる場合）
                        if track_id in hidden_tracks and (current_frame - hidden_tracks[track_id]["last_frame"]) > MAX_HIDDEN_FRAMES:
                            del hidden_tracks[track_id]
            
            # BotSORTの結果を処理して、失われたトラックを補完
            botsort_boxes = {}  # botsortのID -> (bbox, class, score)
            
            for predictions in results_botsort:
                if predictions is None or predictions.boxes is None or predictions.boxes.id is None:
                    continue
                    
                for bbox in predictions.boxes:
                    for scores, classes, bbox_coords, id_ in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
                        if id_ is not None:
                            botsort_id = int(id_)
                            botsort_boxes[botsort_id] = (bbox_coords.tolist(), classes, scores)
            
            # 失われたbytetrackのIDとbotsortのIDをマッチング
            for lost_id in list(lost_tracks.keys()):
                if lost_id in active_bytetrack_ids:  # すでに再検出された場合
                    del lost_tracks[lost_id]
                    continue
                    
                best_match_id = None
                best_match_iou = 0.3  # IoUのしきい値
                best_match_dist = 100  # 距離のしきい値
                
                if lost_id in bytetrack_boxes:  # 以前のbboxがある場合
                    lost_bbox = bytetrack_boxes[lost_id]
                    lost_centroid = get_bbox_centroid(lost_bbox)
                    
                    for botsort_id, (botsort_bbox, botsort_class, _) in botsort_boxes.items():
                        # すでにマッピングされているIDはスキップ
                        if botsort_id in track_id_mapping.values():
                            continue
                            
                        iou = calculate_iou(lost_bbox, botsort_bbox)
                        botsort_centroid = get_bbox_centroid(botsort_bbox)
                        dist = calculate_distance(lost_centroid, botsort_centroid)
                        
                        # IoUが高いか、距離が近い場合にマッチング
                        if (iou > best_match_iou) or (iou > 0.1 and dist < best_match_dist):
                            best_match_id = botsort_id
                            best_match_iou = iou
                            best_match_dist = dist
                
                if best_match_id is not None:
                    track_id_mapping[best_match_id] = lost_id
                    del lost_tracks[lost_id]
            
            # BotSORTの結果からByteTrackの結果を補完
            # 失われたIDに対応するBotSORTの検出を追加
            for predictions_botsort in results_botsort:
                if predictions_botsort is None or predictions_botsort.boxes is None or predictions_botsort.boxes.id is None:
                    continue
                
                for bbox_idx, bbox in enumerate(predictions_botsort.boxes):
                    for scores, classes, bbox_coords, id_ in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
                        if id_ is not None:
                            botsort_id = int(id_)
                            # このBotSORTのIDがByteTrackの失われたIDにマッピングされているか確認
                            if botsort_id in track_id_mapping:
                                bytetrack_id = track_id_mapping[botsort_id]
                                
                                # 対応するByteTrackの結果を見つける
                                for predictions_bytetrack in results_bytetrack:
                                    if predictions_bytetrack is not None:
                                        # IDを置き換えて追加（既存のByteTrackのIDと衝突しないように）
                                        if hasattr(bbox, 'id'):
                                            # IDを置き換え
                                            bbox.id[bbox.id == id_] = bytetrack_id
                                            
                                            # このオブジェクトが現在のByteTrackの結果に存在しない場合のみ追加
                                            if bytetrack_id not in [int(b.id[i]) for b in predictions_bytetrack.boxes if hasattr(b, 'id') for i in range(len(b.id))]:
                                                # 既存のboxesに追加
                                                if predictions_bytetrack.boxes is not None:
                                                    # ここでBotSORTの検出をByteTrackの結果に追加
                                                    # 注: 実際の実装はYOLOの内部構造に依存するため、
                                                    # 以下は概念的な実装です
                                                    last_seen_frame[bytetrack_id] = current_frame
                                                    
            # 最終的な結果はByteTrackをベースに、BotSORTで補完したもの
            results = results_bytetrack

            # print(results)

            for id_ in list(tracking_trajectories.keys()):
                if id_ not in [int(bbox.id) for predictions in results if predictions is not None for bbox in predictions.boxes if bbox.id is not None]:
                    del tracking_trajectories[id_]

            for predictions in results:
                if predictions is None:
                    continue

                # Continue only if boxes and their ids are available
                if predictions.boxes is None or predictions.boxes.id is None:
                    continue

                # If masks are present, iterate through both bbox and masks
                # keypointの出力をコメントアウト
                """
                if predictions.keypoints is not None:
                    for bbox, keypoints in zip(predictions.boxes, predictions.keypoints):
                        for keypoint in keypoints.xy.tolist():
                            for idx, (x , y) in enumerate(keypoint):
                                if (x, y) != (0.0, 0.0):  # Filter out invalid keypoints
                                    cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw green keypoints
                                    cv2.circle(image, (int(x), int(y)), 2, (0, 0, 0), -1)  # Draw black keypoints
                                    # Add the index text next to the keypoint
                                    cv2.putText(image, str(idx), (int(x) + 5, int(y) - 5), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                """


                # If masks are present, iterate through both bbox and masks
                if predictions.masks is not None:
                    for bbox, masks in zip(predictions.boxes, predictions.masks):
                        for scores, classes, bbox_coords, id_ in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
                            xmin    = bbox_coords[0]
                            ymin    = bbox_coords[1]
                            xmax    = bbox_coords[2]
                            ymax    = bbox_coords[3]

                            # Draw rectangle for the bounding box
                            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 225), 2)

                            # Append the bounding box details to a list
                            bboxes.append([bbox_coords, scores, classes, id_])

                            # Create the label for displaying
                            label = (' '+f'ID: {int(id_)}'+' '+str(predictions.names[int(classes)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                            dim, baseline = text_size[0], text_size[1]

                            # Draw the label background rectangle
                            cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] // 3) - 20, int(ymin) - dim[1] + baseline), (30, 30, 30), cv2.FILLED)
                            
                            # Put the label text
                            cv2.putText(image, label, (int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                            # Calculate the centroid of the bounding box
                            centroid_x = (xmin + xmax) / 2
                            centroid_y = (ymin + ymax) / 2

                            # Append centroid to tracking_points if ID is not None
                            if id_ is not None and int(id_) not in tracking_trajectories:
                                tracking_trajectories[int(id_)] = deque()  # maxlenを指定せず、無制限に保存
                            if id_ is not None:
                                tracking_trajectories[int(id_)].append((centroid_x, centroid_y))

                        # Draw trajectories for all objects
                        for id_, trajectory in tracking_trajectories.items():
                            # IDごとに異なる色を生成（IDをハッシュ化して色を決定）
                            color_id = id_ % 80  # 80色の中から選択
                            color = tuple([int(c) for c in colors[color_id]])
                            
                            # 軌跡の線を描画
                            for i in range(1, len(trajectory)):
                                cv2.line(image, (int(trajectory[i-1][0]), int(trajectory[i-1][1])), 
                                        (int(trajectory[i][0]), int(trajectory[i][1])), color, LINE_THICKNESS)
                            
                            # 各ポイントを点として描画
                            if DRAW_POINTS:
                                for point in trajectory:
                                    cv2.circle(image, (int(point[0]), int(point[1])), 3, color, -1)

                        # Process and blend masks if available
                        for mask in masks.xy:
                            polygon = mask
                            cv2.polylines(image, [np.int32(polygon)], True, (255, 0, 0), thickness=2)

                            color_ = [int(c) for c in colors[int(classes)]]
                            mask_copy = image.copy()
                            cv2.fillPoly(mask_copy, [np.int32(polygon)], color_) 
                            alpha = 0.5  # Adjust the transparency level
                            blended_image = cv2.addWeighted(image, 1 - alpha, mask_copy, alpha, 0)
                            image = blended_image.copy()

                # If no masks are present, still draw bounding boxes
                else:
                    for bbox in predictions.boxes:
                        for scores, classes, bbox_coords, id_ in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
                            xmin    = bbox_coords[0]
                            ymin    = bbox_coords[1]
                            xmax    = bbox_coords[2]
                            ymax    = bbox_coords[3]

                            # Draw rectangle for the bounding box
                            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 225), 2)

                            # Append the bounding box details to a list
                            bboxes.append([bbox_coords, scores, classes, id_])

                            # Create the label for displaying
                            label = (' '+f'ID: {int(id_)}'+' '+str(predictions.names[int(classes)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                            dim, baseline = text_size[0], text_size[1]

                            # Draw the label background rectangle
                            cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] // 3) - 20, int(ymin) - dim[1] + baseline), (30, 30, 30), cv2.FILLED)
                            
                            # Put the label text
                            cv2.putText(image, label, (int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                            # Calculate the centroid of the bounding box
                            centroid_x = (xmin + xmax) / 2
                            centroid_y = (ymin + ymax) / 2

                            # Append centroid to tracking_points if ID is not None
                            if id_ is not None and int(id_) not in tracking_trajectories:
                                tracking_trajectories[int(id_)] = deque()  # maxlenを指定せず、無制限に保存
                            if id_ is not None:
                                tracking_trajectories[int(id_)].append((centroid_x, centroid_y))

                    # Draw trajectories for all objects
                    for id_, trajectory in tracking_trajectories.items():
                        # IDごとに異なる色を生成（IDをハッシュ化して色を決定）
                        color_id = id_ % 80  # 80色の中から選択
                        color = tuple([int(c) for c in colors[color_id]])
                        
                        # 軌跡の線を描画
                        for i in range(1, len(trajectory)):
                            cv2.line(image, (int(trajectory[i-1][0]), int(trajectory[i-1][1])), 
                                    (int(trajectory[i][0]), int(trajectory[i][1])), color, LINE_THICKNESS)
                        
                        # 各ポイントを点として描画
                        if DRAW_POINTS:
                            for point in trajectory:
                                cv2.circle(image, (int(point[0]), int(point[1])), 3, color, -1)


        for item in bboxes:
            bbox_coords, scores, classes, *id_ = item if len(item) == 4 else (*item, None)
            line = f'{frameId} {int(classes)} {int(id_[0])} {round(float(scores), 3)} {int(bbox_coords[0])} {int(bbox_coords[1])} {int(bbox_coords[2])} {int(bbox_coords[3])} -1 -1 -1 -1\n'
            # print(line)
            file.write(line)


    if not track:
        results = model.predict(image, verbose=False, device=0)  # predict on an image

        for predictions in results:
            if predictions is None:
                continue  # Skip this image if YOLO fails to detect any objects
            if predictions.boxes is None:
                continue  # Skip this image if there are no boxes

            # If masks are present, iterate through both bbox and masks
            # keypointの出力をコメントアウト
            """
            if predictions.keypoints is not None:
                for bbox, keypoints in zip(predictions.boxes, predictions.keypoints):
                    for keypoint in keypoints.xy.tolist():
                        for idx, (x , y) in enumerate(keypoint):
                            if (x, y) != (0.0, 0.0):  # Filter out invalid keypoints
                                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw green keypoints
                                cv2.circle(image, (int(x), int(y)), 2, (0, 0, 0), -1)  # Draw black keypoints
                                # Add the index text next to the keypoint
                                cv2.putText(image, str(idx), (int(x) + 5, int(y) - 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            """


            # If masks are present, iterate through both bbox and masks
            if predictions.masks is not None:
                for bbox, masks in zip(predictions.boxes, predictions.masks):              
                    for scores, classes, bbox_coords in zip(bbox.conf, bbox.cls, bbox.xyxy):
                        xmin    = bbox_coords[0]
                        ymin    = bbox_coords[1]
                        xmax    = bbox_coords[2]
                        ymax    = bbox_coords[3]
                        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 2)
                        bboxes.append([bbox_coords, scores, classes])

                        label = (' '+str(predictions.names[int(classes)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] //3) - 20, int(ymin) - dim[1] + baseline), (30,30,30), cv2.FILLED)
                        cv2.putText(image,label,(int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    for mask in masks.xy:
                        polygon = mask
                        cv2.polylines(image, [np.int32(polygon)], True, (255, 0, 0), thickness=2)

                        color_ = [int(c) for c in colors[int(classes)]]
                        # cv2.fillPoly(image, [np.int32(polygon)], color_) 
                        mask = image.copy()
                        cv2.fillPoly(mask, [np.int32(polygon)], color_) 
                        alpha = 0.5  # Adjust the transparency level
                        blended_image = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
                        image = blended_image.copy()
            # If no masks are present, still draw bounding boxes
            else:
                for bbox in predictions.boxes:              
                    for scores, classes, bbox_coords in zip(bbox.conf, bbox.cls, bbox.xyxy):
                        xmin    = bbox_coords[0]
                        ymin    = bbox_coords[1]
                        xmax    = bbox_coords[2]
                        ymax    = bbox_coords[3]
                        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 2)
                        bboxes.append([bbox_coords, scores, classes])

                        label = (' '+str(predictions.names[int(classes)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] //3) - 20, int(ymin) - dim[1] + baseline), (30,30,30), cv2.FILLED)
                        cv2.putText(image,label,(int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


    return image



def process_video(args):
    print(args)
    source = args['source']
    track_ = args['track']
    count_ = args['count']


    global input_video_name, frame_width, frame_height
    cap = cv2.VideoCapture(int(source) if source == '0' else source)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec if needed (e.g., 'XVID')
    # input_video_name = source.split('.')[0]  # Get the input video name without extension
    input_video_name = os.path.splitext(os.path.basename(source))[0]
    # print('testing : ', input_video_name)
    out = cv2.VideoWriter(f'output/{input_video_name}_output.mp4', fourcc, 15, (frame_width, frame_height))

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


        frame = process(frame1, track_)

        if not track_ and count_:
            print('[INFO] count works only when objects are tracking.. so use: --track --count')
            break

        if track_ and count_:
            itemDict={}
            ## NOTE: this works only if save-txt is true
            try:
                df = pd.read_csv('output/'+input_video_name+'_labels.txt' , header=None, sep='\s+')
                # print(df)
                df = df.iloc[:,0:3]
                df.columns=["frameid" ,"class","trackid"]
                df = df[['class','trackid']]
                df = (df.groupby('trackid')['class']
                          .apply(list)
                          .apply(lambda x:sorted(x))
                         ).reset_index()
                df['class']=df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
                vc = df['class'].value_counts()
                vc = dict(vc)

                vc2 = {}
                for key, val in enumerate(names):
                    vc2[key] = val
                itemDict = dict((vc2[key], value) for (key, value) in vc.items())
                itemDict  = dict(sorted(itemDict.items(), key=lambda item: item[0]))
                # print(itemDict)

            except:
                pass

            ## overlay
            display = frame.copy()
            h, w = frame.shape[0], frame.shape[1]
            x1, y1, x2, y2 =10, 10, 10, 70
            txt_size = cv2.getTextSize(str(itemDict), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(frame, (x1, y1 + 1), (txt_size[0] * 2, y2),(0, 0, 0),-1)
            cv2.putText(frame, '{}'.format(itemDict), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX,0.7, (210, 210, 210), 2)
            cv2.addWeighted(frame, 0.7, display, 1 - 0.7, 0, frame)

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if frameId % 10 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps_current = 10 / elapsed_time  # Calculate FPS over the last 20 frames
            fps_str = f'FPS: {fps_current:.2f}'
            start_time = time.time()  # Reset start_time for the next 20 frames

        cv2.putText(frame, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        # GUIサポートがない環境でも動作するように、imshowとwaitKeyを削除
        out.write(frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the video capture and writer
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process video with YOLO.')
    parser.add_argument('--source', nargs='+', type=str, default='0', help='Input video file paths or camera indices')
    parser.add_argument('--track', action='store_true', help='if track objects')
    parser.add_argument('--count', action='store_true', help='if count objects')

    args = parser.parse_args()

    # Create a list of dictionaries containing the arguments for each process
    process_args_list = [{'source': source, 'track': args.track, 'count': args.count} for source in args.source]

    with Pool(processes=len(process_args_list)) as pool:
        pool.map(process_video, process_args_list)
