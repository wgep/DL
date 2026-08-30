"""
Batch MediaPipe Export for MotionBERT
Exports Halpe26-format JSON from recorded videos
Authors: Kornel Lipka, Yu-Cian Huang, Ssu-Cheng Chen
Course: Deep Learning and Decision Making, TUM SS2026
"""

import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mediapipe_to_halpe26_json import build_alphapose_json

# Config - EDIT THESE PATHS
# change your path
VIDEOS_FOLDER = r"path to your recorded videos folder"
# change your path
MODEL_FILE    = r"path to pose_landmarker_full.task"
# change your path
OUT_FOLDER    = r"path to save exported JSON files"

VIDEO_FILES = [
    "01_preparation.mp4",
    "02_grasp_birds_tail.mp4",
    "03_single_whip.mp4",
    "04_lift_hand.mp4",
    "05_white_crane.mp4",
    "06_brush_knee.mp4",
    "07_hold_lute.mp4",
    "08_pulling_blocking.mp4",
    "09_apparent_close.mp4",
    "10_cross_hands.mp4",
]


def process_video(video_path, landmarker, out_json_path):
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open: {video_path}")
        return None

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_frame_landmarks = []
    frame_count = 0
    detected_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            all_frame_landmarks.append(result.pose_landmarks[0])
            detected_count += 1
        else:
            all_frame_landmarks.append(None)

    cap.release()

    num_written = build_alphapose_json(all_frame_landmarks, width, height, out_json_path)
    print(f"  {frame_count} frames, {detected_count} detected, wrote {num_written}")
    return num_written


def main():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_FILE)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    os.makedirs(OUT_FOLDER, exist_ok=True)

    for video_file in VIDEO_FILES:
        video_path = os.path.join(VIDEOS_FOLDER, video_file)
        if not os.path.exists(video_path):
            print(f"SKIP (not found): {video_file}")
            continue

        out_json = os.path.join(OUT_FOLDER, video_file.replace(".mp4", "_halpe26.json"))
        print(f"Processing: {video_file}")
        process_video(video_path, landmarker, out_json)

    landmarker.close()
    print("\nAll done. JSON files written to:", OUT_FOLDER)
    print("\nNext: run infer_wild.py on each JSON (see run_all_motionbert.ps1)")


if __name__ == "__main__":
    main()
