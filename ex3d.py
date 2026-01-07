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
# DTW用：キーポイント保存
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

A = [k for k in key_list1 if k is not None]
B = [k for k in key_list2 if k is not None]

def pose_dist(a, b):
    return np.linalg.norm(a - b) / a.shape[0]

distance, path = fastdtw(A, B, dist=pose_dist)

path_dict = {}
for i, j in path:
    if i not in path_dict:
        path_dict[i] = j

new_path = [(i, path_dict[i]) for i in range(len(A))]


############################################
# 再生＆同期表示
############################################
cap1 = cv2.VideoCapture(video_path1)

def draw_skeleton(img, keypoints, color):
    if keypoints is None:
        return

    h, w = img.shape[:2]

    # 点
    for i, (x, y) in enumerate(keypoints):
        if i >= 5 and 0 < x < w and 0 < y < h:
            cv2.circle(img, (int(x), int(y)), 4, color, -1)

    # 線
    for a, b in bones:
        x1, y1 = keypoints[a]
        x2, y2 = keypoints[b]
        if (
            0 < x1 < w and 0 < y1 < h and
            0 < x2 < w and 0 < y2 < h
        ):
            cv2.line(img, (int(x1), int(y1)),
                          (int(x2), int(y2)), color, 2)


for i, j in new_path:
    ret1, frame1 = cap1.read()
    if not ret1:
        break

    k1 = A[i]
    k2 = B[j]

    p1_x, p1_y = hip(k1)
    p2_x, p2_y = hip(k2)

    dx = p1_x - p2_x
    dy = p1_y - p2_y

    k2_shift = k2.copy()
    k2_shift[:, 0] += dx
    k2_shift[:, 1] += dy

    canvas = np.zeros_like(frame1)

    draw_skeleton(canvas, k1, (0, 0, 255))      # 赤：Video1
    draw_skeleton(canvas, k2_shift, (255, 0, 0)) # 青：Video2


    cv2.imshow("DTW Synchronized Pose", canvas)

    if cv2.waitKey(30) == 27:
        break

cap1.release()
cv2.destroyAllWindows()
