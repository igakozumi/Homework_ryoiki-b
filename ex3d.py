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

# 欠損除外
A = [k for k in key_list1 if k is not None]
B = [k for k in key_list2 if k is not None]

def pose_dist(a, b):
    return np.linalg.norm(a - b) / a.shape[0]

distance, path = fastdtw(A, B, dist=pose_dist)

path_dict={}
for i,j in path:
    if i not in path_dict:
        path_dict[i]=j

print(path_dict)
new_path=[]

print(path_dict)
for i in range(len(A)):
    new_path.append((i,path_dict[i]))


############################################
# ★ 再読み込みして同期表示フェーズへ
############################################
cap1 = cv2.VideoCapture(video_path1)
# Note: cap2は直接フレームを読み出すのではなく、保存したリストBからデータを取得します

def draw_skeleton(img, keypoints, color):
    # keypointsがNoneの場合はスキップ
    if keypoints is None: return
    for i, (x, y) in enumerate(keypoints):
        if i >= 5: # 顔以外の主要関節
            cv2.circle(img, (int(x), int(y)), 4, color, -1)
    for a, b in bones:
        x1, y1 = keypoints[a]
        x2, y2 = keypoints[b]
        if (x1, y1) != (0, 0) and (x2, y2) != (0, 0): # 座標が存在する場合
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

# new_path は [(0, j0), (1, j1), (2, j2), ...] という形式
for i, j in new_path:
    ret1, frame1 = cap1.read()
    if not ret1:
        break

    # A[i] は現在のVideo 1の姿勢、B[j] はDTWで対応付けられたVideo 2の姿勢
    k1 = A[i]
    k2 = B[j]

    # ヒップ位置（重心）を取得して位置を合わせる（オーバーレイ用）
    p1_x, p1_y = hip(k1)
    p2_x, p2_y = hip(k2)

    dx = p1_x - p2_x
    dy = p1_y - p2_y

    k2_shift = k2.copy()
    k2_shift[:, 0] += dx
    k2_shift[:, 1] += dy

    # 黒背景のキャンバス、またはframe1をコピー
    canvas = np.zeros_like(frame1)
    # もし実際の映像の上に重ねたい場合は canvas = frame1.copy() に変更してください

    # スケルトン描画
    draw_skeleton(canvas, k1, (0, 0, 255))      # 赤: Video 1
    draw_skeleton(canvas, k2_shift, (255, 0, 0)) # 青: Video 2 (同期)

    # テキストでフレーム番号を表示（デバッグ用）
    cv2.putText(canvas, f"V1 Frame: {i} <-> V2 Frame: {j}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("DTW Synchronized Pose", canvas)

    if cv2.waitKey(30) == 27: # ESCキーで終了
        break

cap1.release()
cv2.destroyAllWindows()