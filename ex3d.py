from ultralytics import YOLO
import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


video_path1 = "ex3a.mp4"
video_path2 = "ex3b.mp4"

cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)

model = YOLO("yolo11x-pose.pt")

bones = [
    (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

def hip(keypoints):
    lx, ly = keypoints[11]
    rx, ry = keypoints[12]
    cx = (lx + rx) / 2
    cy = (ly + ry) / 2    
    return cx, cy


############################################
# ★ DTW追加：全フレームのキーポイント先取り保存
############################################
key_list1 = []
key_list2 = []

while True:
    s1, f1 = cap1.read()
    s2, f2 = cap2.read()
    if not s1 and not s2:
        break

    if s1:
        r1 = model(f1)
        if r1[0].keypoints is not None and r1[0].keypoints.xy.numel() > 0:
            key_list1.append(r1[0].keypoints.xy[0].cpu().numpy())
        else:
            key_list1.append(None)
    if s2:
        r2 = model(f2)
        if r2[0].keypoints is not None and r2[0].keypoints.xy.numel() > 0:
            key_list2.append(r2[0].keypoints.xy[0].cpu().numpy())
        else:
            key_list2.append(None)

cap1.release()
cap2.release()

# 欠損は除外してDTWへ
A = [k for k in key_list1 if k is not None]
B = [k for k in key_list2 if k is not None]

def pose_dist(a, b):
    return np.linalg.norm(a - b)

# 距離行列
dist = [[pose_dist(a, b) for b in B] for a in A]

# DTW実行
disance, path = fastdtw(A, B, dist=euclidean)

print(path)

idxA = alignment.index1
idxB = alignment.index2

############################################
# ★ 再読み込みして同期表示フェーズへ
############################################
cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)

frame_count = 0

def draw_skeleton(img, keypoints, color):
    for i, (x, y) in enumerate(keypoints):
        if i >= 5:
            cv2.circle(img, (int(x), int(y)), 4, color, -1)
    for a, b in bones:
        x1, y1 = keypoints[a]
        x2, y2 = keypoints[b]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

while True:
    ret1, frame1 = cap1.read()
    if not ret1 or frame_count >= len(idxA):
        break

    k1 = A[idxA[frame_count]]
    k2 = B[idxB[frame_count]]

    people1_x, people1_y = hip(k1)
    people2_x, people2_y = hip(k2)

    dx = people1_x - people2_x
    dy = people1_y - people2_y

    k2_shift = k2.copy()
    k2_shift[:, 0] += dx
    k2_shift[:, 1] += dy

    black = np.zeros_like(frame1)

    draw_skeleton(black, k1, (0, 0, 255))  # 赤
    draw_skeleton(black, k2_shift, (255, 0, 0))  # 青

    cv2.imshow("ex3c_dtw", black)

    if cv2.waitKey(20) == 27:
        break

    frame_count += 1

cap1.release()
cap2.release()
cv2.destroyAllWindows()