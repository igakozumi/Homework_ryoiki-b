from ultralytics import YOLO
import cv2
import numpy as np

video_path = "ex2.mp4"

cap = cv2.VideoCapture(video_path)

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

while cap.isOpened():
    success, frame = cap.read()
    # フレームを読み出す

    if success:

        results = model(frame)

        keypoints = results[0].keypoints.xy[0]  # (17, 3)

        # キーポイント描画
        for i, (x, y) in enumerate(keypoints):
            if i >= 5:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)
        
        # ボーン描画
        for (a, b) in bones:
            x1, y1 = keypoints[a][0] ,keypoints[a][1]
            x2, y2 = keypoints[b][0] ,keypoints[b][1]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        cv2.imshow("ex2", frame)

        if cv2.waitKey(20) == 27:
            break
        # ESCが押されれば終了
    else:
        break

cap.release()
cv2.destroyAllWindows()