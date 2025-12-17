import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

results = model.predict("ex4-25.jpg", conf=0.1)

# 入力画像
img = results[0].orig_img

# 認識した物体領域を取得する．
boxes = results[0].boxes

for box in boxes:
    # 物体領域のxy座標を取得する．
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        thickness=3,
    )

cv2.imshow("ex4", img)
cv2.waitKey(0)
cv2.destroyAllWindows()