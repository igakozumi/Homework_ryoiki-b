import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# モデル & 画像読み込み
# =========================
model = YOLO("yolov8n.pt")

src_img = cv2.imread("ex4-25.jpg")
court_img = cv2.imread("soccer_field.png")

if src_img is None or court_img is None:
    raise FileNotFoundError("画像が読み込めません")

# =========================
# 射影変換用対応点
# （適当に選んでよい）
# =========================
pts_src = np.array([
    (955, 71),
    (702, 150),
    (1080, 120),
    (1156, 422)
], dtype=np.float32)

pts_ref = np.array([
    (3150, 229),
    (2715, 589),
    (3147, 593),
    (2734, 1633)
], dtype=np.float32)

H = cv2.getPerspectiveTransform(pts_src, pts_ref)

# =========================
# 人検出（YOLO）
# =========================
results = model.predict(src_img, conf=0.1)

person_boxes = []
for result in results:
    for box in result.boxes:
        if int(box.cls[0]) == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_boxes.append((x1, y1, x2, y2))

# =========================
# コート画像（縮小）
# =========================
scale = 0.3
field_small = cv2.resize(
    court_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
)

h, w = field_small.shape[:2]

# =========================
# 足元代表点 → 射影変換 → 描画
# =========================
for x1, y1, x2, y2 in person_boxes:
    # 足元代表点
    cx = int((x1 + x2) / 2)
    cy = y2

    pt = np.array([[[cx, cy]]], dtype=np.float32)
    pt_w = cv2.perspectiveTransform(pt, H)

    X, Y = pt_w[0, 0]

    # 縮小画像に合わせる
    Xs = int(X * scale)
    Ys = int(Y * scale)

    # # =========================
    # # コート内判定（枠線外除外）
    # # =========================
    # if 50 < Xs < w - 50 and 50 < Ys < h - 50:
    #     cv2.circle(
    #         field_small,
    #         (Xs, Ys),
    #         10,
    #         (255, 0, 255),  # ピンク
    #         2               # 縁のみ
    #     )

# =========================
# コート内判定（Hの出力座標系）
# =========================
COURT_X_MIN = 300
COURT_X_MAX = 3300
COURT_Y_MIN = 200
COURT_Y_MAX = 1800

for x1, y1, x2, y2 in person_boxes:
    # 足元代表点
    cx = int((x1 + x2) / 2)
    cy = y2

    pt = np.array([[[cx, cy]]], dtype=np.float32)
    pt_w = cv2.perspectiveTransform(pt, H)

    X, Y = pt_w[0, 0]

    # ★ ここで判定（射影後座標）
    if (COURT_X_MIN < X < COURT_X_MAX and
        COURT_Y_MIN < Y < COURT_Y_MAX):

        # 縮小画像に合わせる
        Xs = int(X * scale)
        Ys = int(Y * scale)

        cv2.circle(
            field_small,
            (Xs, Ys),
            10,
            (255, 0, 255),  # ピンク
            2
        )



# =========================
# 表示
# =========================
cv2.imshow("Players on court", field_small)
cv2.waitKey(0)
cv2.destroyAllWindows()