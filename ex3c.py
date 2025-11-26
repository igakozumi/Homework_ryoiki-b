from ultralytics import YOLO
import cv2
import numpy as np

video_path1 = "ex3a.mp4"
video_path2 = "ex3b.mp4"

cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)

# モデル読み込み（pose用）
model = YOLO("yolo11x-pose.pt")

# ボーン（COCO17用）
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

# 動画を開く
cnt = 0

# フレーム番号を0にする

while True:
    success1, frame1 = cap1.read()
    success2, frame2 = cap2.read()

    # どちらかが終わったら、終わった側は黒フレーム（サイズ合わせ）にする
    if not success1:
        frame1 = np.zeros_like(frame2)
    if not success2:
        frame2 = np.zeros_like(frame1)

    results1 = model(frame1)
    results2 = model(frame2)

    keypoints1 = results1[0].keypoints.xy[0].cpu().numpy()  # (17, 3)
    keypoints2 = results2[0].keypoints.xy[0].cpu().numpy()  # (17, 3)

    people1_x, people1_y = hip(keypoints1)
    people2_x, people2_y = hip(keypoints2)

    dx = people1_x - people2_x
    dy = people1_y - people2_y

    keypoints2_shifted = keypoints2.copy()
    keypoints2_shifted[:, 0] += dx
    keypoints2_shifted[:, 1] += dy

    #背景変更
    black = np.zeros_like(frame1)

    # キーポイント描画

    for i, (x, y) in enumerate(keypoints1):
        if i >= 5:
            cv2.circle(black, (int(x), int(y)), 4, (0, 255, 255), -1)
    
    for i, (x, y) in enumerate(keypoints2_shifted):
            if i >= 5:
                cv2.circle(black, (int(x), int(y)), 4, (0, 255, 255), -1)


    # ボーン描画
    for (a, b) in bones:
            x1, y1 = keypoints1[a][0] ,keypoints1[a][1]
            x2, y2 = keypoints1[b][0] ,keypoints1[b][1]
            cv2.line(black, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    for (a, b) in bones:
            x1, y1 = keypoints2_shifted[a][0] ,keypoints2_shifted[a][1]
            x2, y2 = keypoints2_shifted[b][0] ,keypoints2_shifted[b][1]
            cv2.line(black, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

   

    cv2.imshow("ex3c", black)

    if cv2.waitKey(20) == 27:
        break


cap1.release()
cap2.release()
cv2.destroyAllWindows()