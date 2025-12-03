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

# 前フレームのキーポイントを保存
keypoints1 = None
keypoints2 = None

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

    if not success1 and not success2:
        break  # 両方終了したら終わり

    # 黒画像を生成
    if not success1:
        frame1 = np.zeros((frame2.shape[0], frame2.shape[1], 3), dtype=np.uint8)
    if not success2:
        frame2 = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)


    results1 = model(frame1)
    results2 = model(frame2)


    #検出成功したら更新、失敗したら前のキーポイントを使用
    if results1[0].keypoints is not None and results1[0].keypoints.xy.numel() > 0:
        keypoints1 = results1[0].keypoints.xy[0].cpu().numpy()
    # 検出失敗
    if results2[0].keypoints is not None and results2[0].keypoints.xy.numel() > 0:
        keypoints2 = results2[0].keypoints.xy[0].cpu().numpy()

    #どっちも無かったら描画できないのでスキップ
    if keypoints1 is None and keypoints2 is None:
        cv2.imshow("ex3b", black)
        if cv2.waitKey(20) == 27:
            break
        continue

    people1_x, people1_y = hip(keypoints1)
    people2_x, people2_y = hip(keypoints2)

    dx = people1_x - people2_x
    dy = people1_y - people2_y

    keypoints2_shifted = keypoints2.copy()
    keypoints2_shifted[:, 0] += dx
    keypoints2_shifted[:, 1] += dy

    #背景変更
    black = np.zeros_like(frame1)

    # 描画関数
    def draw_skeleton(keypoints, color):
        for i, (x, y) in enumerate(keypoints):
            if i >= 5:
                cv2.circle(black, (int(x), int(y)), 4, (0, 255, 255), -1)
        for a, b in bones:
            x1, y1 = keypoints[a]
            x2, y2 = keypoints[b]
            cv2.line(black, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


    if keypoints1 is not None:
        draw_skeleton(keypoints1, (0, 0, 255))  # 赤
    if keypoints2_shifted is not None:
        draw_skeleton(keypoints2_shifted, (255, 0, 0))  # 青

    cv2.imshow("ex3c", black)

    if cv2.waitKey(20) == 27:
        break


cap1.release()
cap2.release()
cv2.destroyAllWindows()