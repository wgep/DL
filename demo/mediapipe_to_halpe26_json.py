"""
MediaPipe to Halpe26 JSON Converter
Converts MediaPipe landmarks to MotionBERT's expected input format
Authors: Kornel Lipka, Yu-Cian Huang, Ssu-Cheng Chen
Course: Deep Learning and Decision Making, TUM SS2026
"""

import json
import numpy as np


def mediapipe_landmarks_to_halpe26(landmarks, width, height):
    """
    landmarks: MediaPipe pose_landmarks (33 landmarks) for ONE frame.
    width, height: pixel dimensions of the source video frame - needed
                   because AlphaPose/infer_wild.py expects PIXEL
                   coordinates, not MediaPipe's 0-1 normalized ones.
    Returns: flat list of 78 values [x0,y0,c0, x1,y1,c1, ... x25,y25,c25]
             ready to go into the 'keypoints' field of the JSON.
    """
    def px(lm):
        conf = getattr(lm, "visibility", 1.0)
        return [lm.x * width, lm.y * height, conf]

    l_sh, r_sh = landmarks[11], landmarks[12]
    l_hip, r_hip = landmarks[23], landmarks[24]

    kpts = [None] * 26
    kpts[0]  = px(landmarks[0])                     # Nose
    kpts[1]  = px(landmarks[2])                      # LEye
    kpts[2]  = px(landmarks[5])                      # REye
    kpts[3]  = px(landmarks[7])                      # LEar
    kpts[4]  = px(landmarks[8])                      # REar
    kpts[5]  = px(l_sh)                               # LShoulder
    kpts[6]  = px(r_sh)                               # RShoulder
    kpts[7]  = px(landmarks[13])                      # LElbow
    kpts[8]  = px(landmarks[14])                      # RElbow
    kpts[9]  = px(landmarks[15])                      # LWrist
    kpts[10] = px(landmarks[16])                      # RWrist
    kpts[11] = px(l_hip)                              # LHip
    kpts[12] = px(r_hip)                              # RHip
    kpts[13] = px(landmarks[25])                      # LKnee
    kpts[14] = px(landmarks[26])                      # RKnee
    kpts[15] = px(landmarks[27])                      # LAnkle
    kpts[16] = px(landmarks[28])                      # RAnkle
    kpts[17] = px(landmarks[0])                       # Head (approx: reuse nose,
                                                        # MediaPipe has no distinct
                                                        # head-top landmark)
    # Neck and Hip are midpoints - build directly, confidence = min of parents
    neck_x = (l_sh.x + r_sh.x) / 2 * width
    neck_y = (l_sh.y + r_sh.y) / 2 * height
    neck_c = min(getattr(l_sh, "visibility", 1.0), getattr(r_sh, "visibility", 1.0))
    kpts[18] = [neck_x, neck_y, neck_c]                # Neck

    hip_x = (l_hip.x + r_hip.x) / 2 * width
    hip_y = (l_hip.y + r_hip.y) / 2 * height
    hip_c = min(getattr(l_hip, "visibility", 1.0), getattr(r_hip, "visibility", 1.0))
    kpts[19] = [hip_x, hip_y, hip_c]                   # Hip

    kpts[20] = px(landmarks[31])                       # LBigToe
    kpts[21] = px(landmarks[32])                       # RBigToe
    kpts[22] = px(landmarks[31])                       # LSmallToe (reuse - MediaPipe
                                                         # has no distinct small toe)
    kpts[23] = px(landmarks[32])                       # RSmallToe (reuse)
    kpts[24] = px(landmarks[29])                       # LHeel
    kpts[25] = px(landmarks[30])                       # RHeel

    flat = []
    for x, y, c in kpts:
        flat.extend([float(x), float(y), float(c)])
    return flat


def build_alphapose_json(all_frame_landmarks, width, height, out_json_path):
    """
    all_frame_landmarks: list, one entry per frame, each entry being the
                          33 MediaPipe landmarks for that frame (or None
                          if no pose was detected in that frame - such
                          frames are skipped, matching how AlphaPose
                          itself would just not report a detection).
    width, height: video frame pixel dimensions.
    out_json_path: where to write the resulting JSON file.
    """
    results = []
    for landmarks in all_frame_landmarks:
        if landmarks is None:
            continue
        kpts_flat = mediapipe_landmarks_to_halpe26(landmarks, width, height)
        results.append({
            "keypoints": kpts_flat,
            "score": 1.0
            # NOTE: 'idx' intentionally omitted - infer_wild.py's
            # read_input() only checks item['idx'] when --focus is
            # passed on the command line. We won't pass --focus, so
            # every entry is used as-is, in order, one per frame.
        })

    with open(out_json_path, "w") as f:
        json.dump(results, f)

    print(f"Wrote {len(results)} frames to {out_json_path}")
    return len(results)


if __name__ == "__main__":
    # Self-test with fake landmarks - verifies shapes/logic only.
    class FakeLandmark:
        def __init__(self, x, y, v=1.0):
            self.x, self.y, self.visibility = x, y, v

    fake_frame = [FakeLandmark(0.5, 0.5) for _ in range(33)]
    flat = mediapipe_landmarks_to_halpe26(fake_frame, width=640, height=360)
    assert len(flat) == 78, f"Expected 78 values, got {len(flat)}"
    print("Self-test passed. 78 values per frame, as expected (26 keypoints x 3).")
