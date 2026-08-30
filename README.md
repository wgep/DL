Fine-Grained Tai Chi Action Recognition from Skeleton Data

A comparative study of BiLSTM and STGCN architectures for classifying
10 Tai Chi actions from 3D skeleton data (HIT605 dataset).

Kornel Lipka, Yu-Cian Huang, Ssu-Cheng Chen
Deep Learning and Decision Making, TUM, Summer Semester 2026


REPOSITORY STRUCTURE

DL/

  main code/   - Core code: train the two models reported in our paper
  
  images/      - Figures used in the written report
 
  videos/      - Reference videos for each Tai Chi movement
  
  demo/        - Additional real-video demo pipeline (not part of the paper)
  
  output/      - Trained model checkpoints


MAIN CODE

The two scripts that produce all results reported in our paper:

- taichi_lstm_final.py - trains the BiLSTM baseline and
  BiLSTM + temporal attention models
- taichi_stgcn_final.py - trains the STGCN model

Both scripts expect the HIT605 TaiChi dataset. Update the path near the
top of each file (# change your path) before running.

Model                  Top-1 Accuracy   Macro F1   Params
BiLSTM (baseline)      89.25%           0.8825     583K
BiLSTM + Attention     88.63%           0.8702     583K
STGCN                  86.40%           0.8506     23K


OUTPUT

Trained model checkpoints:

- best_without_attention.pth - BiLSTM baseline
- best_with_attention.pth - BiLSTM + temporal attention
- best_stgcn.pth - STGCN
- best_with_attention_augmented.pth - BiLSTM + attention, trained
  with data augmentation (used as the starting point for the demo
  pipeline's fine-tuning step)
- best_with_attention_finetuned.pth - the augmented model further
  fine-tuned on real recorded video (demo pipeline only, not used
  for the paper's reported results)

DEMO (additional, not part of the written report)

This folder contains an extra pipeline we built to test the trained
model on real recorded video, using MediaPipe and MotionBERT for pose
estimation, plus fine-tuning on our own footage. This work is shown in
our presentation video but is NOT part of the formal written report.


- mediapipe_to_halpe26_json.py - converts MediaPipe landmarks to
  MotionBERT's expected input format
- batch_export_halpe26.py - runs MediaPipe on recorded videos
- run_all_motionbert.ps1 - runs MotionBERT on the exported data
- motionbert_joint_mapping.py - converts MotionBERT's H36M joint
  order to our HIT605 joint order
- finetune_on_real_videos.py - fine-tunes the model on real footage
- test_finetuned_all_videos.py - final per-video accuracy test
- build_final_demo_video.py - builds the demo video shown in our
  presentation

Run order:
1. batch_export_halpe26.py (normal Python env, needs mediapipe/opencv)
   -> produces Halpe26 JSON files from your recorded videos
2. run_all_motionbert.ps1 (inside a MotionBERT install + its own venv)
   -> produces X3D.npy 3D pose output for each video (slow, 1-3 hours)
3. finetune_on_real_videos.py (normal Python env)
   -> fine-tunes the model on the real-video data
4. test_finetuned_all_videos.py
   -> reports final per-video accuracy
5. build_final_demo_video.py
   -> builds the demo video shown in our presentation

Note: running this pipeline requires MotionBERT installed separately
(https://github.com/Walter0807/MotionBERT), which is not included in
this repo.

IMAGES

Figures used in the written report (confusion matrices, training
curves, attention weights, class distribution, pipeline overview,
skeleton graph).


VIDEOS

Reference videos for each of the 10 Tai Chi movements.
