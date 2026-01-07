# import cv2
# import numpy as np

# # ===============================
# # 画像を画面内に収める
# # ===============================
# def resize_for_display(img, max_size=900):
#     h, w = img.shape[:2]
#     scale = min(max_size / w, max_size / h, 1.0)
#     resized = cv2.resize(img, (int(w * scale), int(h * scale)))
#     return resized, scale

# ===============================
# 4点クリック取得（スケール補正あり）
# ===============================
# def get_four_points(img, window_name):
#     img_disp, scale = resize_for_display(img)
#     points = []

#     def mouse_callback(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             orig_x = int(x / scale)
#             orig_y = int(y / scale)
#             points.append((orig_x, orig_y))
#             print(f"{window_name} clicked: ({orig_x}, {orig_y})")

#             cv2.circle(img_disp, (x, y), 8, (255, 0, 0), -1)
#             cv2.imshow(window_name, img_disp)

#             if len(points) == 4:
#                 cv2.destroyAllWindows()

#     cv2.imshow(window_name, img_disp)
#     cv2.setMouseCallback(window_name, mouse_callback)
#     cv2.waitKey(0)

#     return np.array(points, dtype=np.float32)

# ===============================
# ① 実写画像（変換元）
# ===============================
# img = cv2.imread("ex4-25.jpg")
# print("【実写画像】左上 → 右上 → 右下 → 左下 の順でクリック")
# pts1 = get_four_points(img, "Input Image")

# # ===============================
# # ② 基準画像（変換先）
# # ===============================
# base = cv2.imread("soccer_field.png")
# print("【基準画像】左上 → 右上 → 右下 → 左下 の順でクリック")
# pts2 = get_four_points(base, "Base Image")

# # ===============================
# # ③ 射影変換
# # ===============================
# M = cv2.getPerspectiveTransform(pts1, pts2)
# h, w = base.shape[:2]
# warped = cv2.warpPerspective(img, M, (w, h))

# # ===============================
# # ④ 射影変換結果のみ表示
# # ===============================
# warped_disp, _ = resize_for_display(warped)
# cv2.imshow("Projected Image", warped_disp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import numpy as np
import matplotlib.pyplot as plt
import cv2

# 基準とするマット四隅の写真上の座標（単位px）
pts1 = np.array([(955, 71), (702, 150), (1080, 120), (1156, 422)], dtype=np.float32)
# 基準とするマット四隅の実際の座標（単位mm）
pts2 = np.array([(3150, 229), (2715, 589), (3147, 593), (2734, 1633)], dtype=np.float32)

# 射影行列の取得
M = cv2.getPerspectiveTransform(pts1, pts2)
np.set_printoptions(precision=5, suppress=True)
print(M)

# 元画像
img1 = cv2.imread("ex4-25.jpg", cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# 元画像を射影変換し鳥瞰画像へ
w2, h2 = pts2.max(axis=0).astype(int) + 10  # 鳥瞰画像サイズを拡張（見た目の調整）
img2 = cv2.warpPerspective(img1, M, (w2, h2))

# 結果表示
plt.imshow(img2)
plt.show()