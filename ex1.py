from ultralytics import YOLO
import cv2
import numpy as np

# 画像を読み込む
img = cv2.imread("ex1.jpg")

# モデル読み込み（pose用）
model = YOLO("yolo11x-pose.pt")

results = model(img)

keypoints = results[0].keypoints.xy[0]  # (17, 3)

# ボーン（COCO17用）
bones = [
    (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# キーポイント描画
for i, (x, y) in enumerate(keypoints):
    if i >= 5:
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

# ボーン描画
for (a, b) in bones:
    x1, y1 = keypoints[a][0] ,keypoints[a][1]
    x2, y2 = keypoints[b][0] ,keypoints[b][1]
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# 出力表示
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存したい場合
cv2.imwrite("ex1_output.jpg", img)