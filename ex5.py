import cv2
from ultralytics import YOLO

# YOLOv8モデルをロード
model = YOLO("yolov8n.pt")

# ビデオファイルを開く
video_path = "ex5.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    # トラッキング実行
    results = model.track(frame,conf = 0.1, persist=True)

    # 検出結果を取得
    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])

            # person クラスのみ（COCOでは 0）
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 赤枠で描画
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

    # フレーム表示
    cv2.imshow("Person Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()