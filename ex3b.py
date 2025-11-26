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

    keypoints1 = results1[0].keypoints.xy[0]  # (17, 3)
    keypoints2 = results2[0].keypoints.xy[0]  # (17, 3)

    #背景変更
    black = np.zeros_like(frame1)

    # キーポイント描画

    def draw_skeleton(keypoints):
        for i, (x, y) in enumerate(keypoints):
            if i >= 5:
                cv2.circle(black, (int(x), int(y)), 4, (0, 255, 255), -1)

    # ボーン描画

        for (a, b) in bones:
            x1, y1 = keypoints[a][0] ,keypoints[a][1]
            x2, y2 = keypoints[b][0] ,keypoints[b][1]
            cv2.line(black, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    draw_skeleton(keypoints1)
    draw_skeleton(keypoints2)

    cv2.imshow("ex3b", black)

    if cv2.waitKey(20) == 27:
        break


cap1.release()
cap2.release()
cv2.destroyAllWindows()