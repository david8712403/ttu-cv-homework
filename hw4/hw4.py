"""
Computer Vision Homework 4
Create by David in 2021/05/10

將測試圖片以Hough transform做直線偵測，將結果圖片印出。
"""

import cv2
import numpy as np

if __name__ == '__main__':
    file = 'input_img.png'
    im = cv2.imread(file, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    """Gaussian blur"""
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    """Canny"""
    low_threshold = 50
    high_threshold = 150
    masked_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    """Hough Transform"""
    rho = 1
    theta = np.pi / 180
    threshold = 1
    min_line_length = 10
    max_line_gap = 1
    line_image = np.copy(im) * 0
    lines = cv2.HoughLinesP(
        image=masked_edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        lines=np.array([]),
        minLineLength=min_line_length,
        maxLineGap=max_line_gap)

    """Draw lines"""
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    """Combine 2 images"""
    color_edges = np.dstack((gray, gray, gray))
    print(f"line_image.shape: {line_image.shape}")
    print(f"color_edges.shape: {color_edges.shape}")
    combine_im = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

    output = np.concatenate((
        im,
        combine_im
    ), axis=1)
    cv2.imshow(file, output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
