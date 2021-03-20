# 作業一：
# 1.自拍一張含自己人像的彩色影像,將其轉為灰階影像
#   列印並說明該灰階影像的檔案格式(JPEG,PNG,BMP?)
#   及大小(行數,列數,位元數)
# 2.將上面含自己人像的灰階影像用方框框出嘴巴
#   用圓框框出眼睛,並列印顯示的結果
#   (以上需說明執行過程與使用OpenCV之程式內容)

import cv2

file = 'image.jpg'
im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
# 框出嘴吧
cv2.rectangle(
    im,
    pt1=(750, 1200),
    pt2=(1250, 1400),
    color=(0, 255, 0),
    thickness=10
)
# 框出眼睛
# 左眼
cv2.circle(
    im,
    center=(650, 960),
    radius=120,
    color=(0, 255, 0),
    thickness=10
)
# 右眼
cv2.circle(
    im,
    center=(1100, 840),
    radius=120,
    color=(0, 255, 0),
    thickness=10
)
cv2.putText(
    im, f"image size: {im.shape}",
    (1200, 100), cv2.FONT_HERSHEY_SIMPLEX,
    2, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imshow(file, im)
cv2.waitKey(0)
cv2.destroyAllWindows()
