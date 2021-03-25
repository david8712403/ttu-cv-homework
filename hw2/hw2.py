"""
Computer Vision Homework 2
1.  將一張含自己人像的灰階影像
    使用OpenCV圖像縮放(resize)函式成128x128影像
2.  將上面128x128影像加上5%鹽和胡椒雜訊
    再分別使用OpenCV平均濾波器(blur)函式和
    中位濾波器(medianBlur)函式去除雜訊，比較結果。
3.  程式執行以處理影像像素運算方式重作(2)
    （自己實作一個中位濾波器，要描述邊緣如何處理）
"""

import cv2
import numpy as np


def addSaltPepper(img, SNR):
    """
    依照SNR(訊雜比)將影像加入椒鹽雜訊
    :param img: 輸入影像
    :param SNR: 訊雜比例
    :return: 輸出影像
    """
    _img = img.copy()
    w, h, c = _img.shape
    mask = np.random.choice((0, 1, 2), size=(w, h), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    for _w in range(0, w):
        for _h in range(0, h):
            if mask[_w][_h] == 0:
                _img[_w][_h] = img[_w][_h]
            elif mask[_w][_h] == 1:
                _img[_w][_h] = (255, 255, 255)
            elif mask[_w][_h] == 2:
                _img[_w][_h] = (0, 0, 0)
    return _img


def mAveBlur(img):
    """
    自己實作一個平均濾波器
    :param img: 輸入影像
    :return: 輸出影像
    """
    _img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_WRAP)
    w, h, c = _img.shape
    for _w in range(1, w - 1):
        for _h in range(1, h - 1):
            for _c in range(0, c):
                roi = _img[_w - 1:_w + 2, _h - 1:_h + 2, _c]
                _img[_w][_h][_c] = np.mean(roi)

    _img = _img[1:w - 1, 1:h - 1, 0:3]
    return _img


def mMedianBlur(img):
    """
    自己實作一個size為3x3的中位濾波器
    :param img: 輸入影像
    :return: 輸出影像
    """
    # 上下左右補上寬度為1的邊框
    _img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT,value=[0,0,0])
    w, h, c = _img.shape
    for _w in range(1, w-1):
        for _h in range(1, h-1):
            for _c in range(0, c):
                roi = _img[_w-1:_w+2, _h-1:_h+2, _c]
                # print(f"before: roi.shape{roi}, median:{_img[_w][_h][_c]}")
                _img[_w][_h][_c] = np.median(roi)
                # print(f"after: roi.shape{roi}")

    _img = _img[1:w-1, 1:h-1, 0:3]
    return _img


def mPutText(img, text):
    cv2.putText(img, str(text), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == '__main__':

    file = 'image.png'
    size = (128, 128)
    im = cv2.imread(file)
    im = cv2.resize(im, size)
    im_add_noise = addSaltPepper(im, 0.95)
    # average blur
    im_ave_blur = cv2.blur(im_add_noise, (3, 3))
    im_my_ave_blur = mAveBlur(im_add_noise)
    # median blur
    im_median_blur = cv2.medianBlur(im_add_noise, 3)
    im_my_filter = mMedianBlur(im_add_noise)

    # put text
    mPutText(im, 'resize (128,128)')
    mPutText(im_add_noise, 'add noise')
    # average blur
    mPutText(im_ave_blur, 'opencv aveBlur')
    mPutText(im_my_ave_blur, 'my aveBlur')
    # median blur
    mPutText(im_median_blur, 'opencv medianBlur')
    mPutText(im_my_filter, 'my medianBlur')

    output = np.concatenate((
        im,
        im_add_noise,
        im_ave_blur,
        im_my_ave_blur,
        im_median_blur,
        im_my_filter
    ),axis=1)
    cv2.imshow(file, output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


