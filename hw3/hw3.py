"""
Computer Vision Homework 3
Create by David in 2021/04/17

將一張含自己人像的彩色影像以Canny或Sobel edge detection做邊緣偵測，並完成：
1. 印出邊緣偵測後之二值影像
2. 印出邊緣偵測後之對應的彩色影像(將非邊緣像素設為全白)
"""

import cv2
import numpy as np


def get_edge_im(image, colorful):
    """
    取得邊緣偵測圖像
    :param image: 輸入影像
    :param colorful: 邊緣是否保留原本色彩
    :return: 邊緣偵測圖像
    """
    h, w, c = image.shape
    kernel_size = 9
    blur_im = cv2.GaussianBlur(
        src=im,
        ksize=(kernel_size, kernel_size),
        sigmaX=-1
    )

    tmp = cv2.Canny(
        blur_im,
        threshold1=50,
        threshold2=100,
    )

    edge_im = np.zeros_like(image)
    if colorful:
        for _w in range(0, w):
            for _h in range(0, h):
                if tmp[_h][_w] == 255:
                    for _c in range(0, c):
                        edge_im[_h][_w][_c] = image[_h][_w][_c]
                else:
                    for _c in range(0, c):
                        edge_im[_h][_w][_c] = 255
    else:
        edge_im[:, :, 0] = 255 - tmp
        edge_im[:, :, 1] = 255 - tmp
        edge_im[:, :, 2] = 255 - tmp

    return edge_im


def mPutText(img, text):
    cv2.rectangle(
        img=img,
        pt1=(0, 0),
        pt2=(200, 20),
        color=(255, 255, 255),
        thickness=-1
    )
    cv2.putText(
        img=img,
        text=str(text),
        org=(0, 15),
        fontFace=cv2.FONT_ITALIC,
        fontScale=0.5,
        color=(0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_4
    )


if __name__ == '__main__':
    file = 'image.jpg'
    im = cv2.imread(file, cv2.COLOR_RGB2GRAY)
    w = int(im.shape[1] * 20 / 100)
    h = int(im.shape[0] * 20 / 100)
    im = cv2.resize(im, dsize=(w, h))
    im_edges = get_edge_im(
        image=im,
        colorful=False
    )
    im_colorful_edges = get_edge_im(
        image=im,
        colorful=True
    )
    mPutText(im, "original")
    mPutText(im_edges, "edge image")
    mPutText(im_colorful_edges, "colorful edge image")

    output = np.concatenate((
        im,
        im_edges,
        im_colorful_edges
    ), axis=1)
    cv2.imshow(file, output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
