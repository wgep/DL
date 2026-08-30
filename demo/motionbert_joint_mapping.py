"""
MotionBERT Joint Mapping
H36M17 <-> HIT605-17 conversion
Authors: Kornel Lipka, Yu-Cian Huang, Ssu-Cheng Chen
Course: Deep Learning and Decision Making, TUM SS2026
"""

import numpy as np


# Step 1: MediaPipe 33 landmarks -> H36M 17 keypoints
# MediaPipe BlazePose landmark indices (standard, 0-32):
#   0 nose, 11 left shoulder, 12 right shoulder, 13 left elbow,
#   14 right elbow, 15 left wrist, 16 right wrist, 23 left hip,
#   24 right hip, 25 left knee, 26 right knee, 27 left ankle,
#   28 right ankle
#
# H36M 17-joint order (what MotionBERT expects as input AND produces
# as output):
#   0 Hip(pelvis), 1 RHip, 2 RKnee, 3 RAnkle, 4 LHip, 5 LKnee,
#   6 LAnkle, 7 Spine, 8 Neck/Thorax, 9 Nose, 10 Head,
#   11 LShoulder, 12 LElbow, 13 LWrist, 14 RShoulder, 15 RElbow,
#   16 RWrist
def mediapipe_to_h36m(landmarks):
    """
    landmarks: MediaPipe pose_landmarks (33 landmarks, each with .x, .y,
               [.z if available], and ideally .visibility as a confidence
               proxy). Use IMAGE-space pose_landmarks here (0-1 normalized
               x,y), NOT pose_world_landmarks - MotionBERT wants 2D input.
    Returns: (17, 3) array - x, y, confidence (z left as 0/visibility).
    """
    coords = np.array([[lm.x, lm.y, getattr(lm, "visibility", 1.0)]
                        for lm in landmarks])  # (33, 3)

    l_shoulder, r_shoulder = coords[11], coords[12]
    l_hip, r_hip           = coords[23], coords[24]
    neck   = (l_shoulder + r_shoulder) / 2
    pelvis = (l_hip + r_hip) / 2

    h36m = np.zeros((17, 3), dtype=np.float32)
    h36m[0]  = pelvis
    h36m[1]  = r_hip
    h36m[2]  = coords[26]              # RKnee
    h36m[3]  = coords[28]              # RAnkle
    h36m[4]  = l_hip
    h36m[5]  = coords[25]              # LKnee
    h36m[6]  = coords[27]              # LAnkle
    h36m[7]  = (neck + pelvis) / 2     # Spine
    h36m[8]  = neck                    # Neck/Thorax
    h36m[9]  = coords[0]               # Nose (approximates H36M's "Nose")
    h36m[10] = coords[0]               # Head - MediaPipe has no distinct
                                        # head-top landmark, so we reuse
                                        # nose here. This is an approximation.
    h36m[11] = l_shoulder
    h36m[12] = coords[13]              # LElbow
    h36m[13] = coords[15]              # LWrist
    h36m[14] = r_shoulder
    h36m[15] = coords[14]              # RElbow
    h36m[16] = coords[16]              # RWrist
    return h36m


# Step 2: H36M 17 -> HIT605 17 (Chest joint synthesized as shoulder midpoint)

def h36m_to_hit605(h36m_joints):
    """
    h36m_joints: (T, 17, 3) - MotionBERT's 3D output for a sequence.
    Returns: (T, 17, 3) in HIT605 joint order, ready for your existing
             normalize_joints() + BiLSTM pipeline.
    """
    T = h36m_joints.shape[0]
    out = np.zeros((T, 17, 3), dtype=np.float32)

    out[:, 0]  = h36m_joints[:, 0]   # Pelvis
    out[:, 1]  = h36m_joints[:, 1]   # RHip
    out[:, 2]  = h36m_joints[:, 2]   # RKnee
    out[:, 3]  = h36m_joints[:, 3]   # RAnkle
    out[:, 4]  = h36m_joints[:, 4]   # LHip
    out[:, 5]  = h36m_joints[:, 5]   # LKnee
    out[:, 6]  = h36m_joints[:, 6]   # LAnkle
    out[:, 7]  = h36m_joints[:, 7]   # Spine
    out[:, 8]  = h36m_joints[:, 8]   # Neck
    out[:, 9]  = h36m_joints[:, 10]  # Head (H36M's "Head", index 10)
    out[:, 10] = h36m_joints[:, 11]  # LShoulder
    out[:, 11] = h36m_joints[:, 12]  # LElbow
    out[:, 12] = h36m_joints[:, 13]  # LWrist
    out[:, 13] = h36m_joints[:, 14]  # RShoulder
    out[:, 14] = h36m_joints[:, 15]  # RElbow
    out[:, 15] = h36m_joints[:, 16]  # RWrist
    out[:, 16] = (h36m_joints[:, 11] + h36m_joints[:, 14]) / 2  # Chest (synthesized)

    return out


# Quick self-test (no MotionBERT needed) 
if __name__ == "__main__":
    class FakeLandmark:
        def __init__(self, x, y, v=1.0):
            self.x, self.y, self.visibility = x, y, v

    fake_landmarks = [FakeLandmark(0.5, 0.5) for _ in range(33)]
    h36m_2d = mediapipe_to_h36m(fake_landmarks)
    print("mediapipe_to_h36m output shape:", h36m_2d.shape)  # (17, 3)

    fake_h36m_3d_sequence = np.random.randn(10, 17, 3).astype(np.float32)
    hit605_seq = h36m_to_hit605(fake_h36m_3d_sequence)
    print("h36m_to_hit605 output shape:", hit605_seq.shape)  # (10, 17, 3)
    print("Self-test passed.")
