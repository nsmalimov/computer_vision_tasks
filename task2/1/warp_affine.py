#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

img = cv2.imread('image.jpg', 0)
rows, cols = img.shape


def window_transform(x_left_top_1, y_left_top_1, x_right_down_1, y_right_down_1, x_left_top_2, y_left_top_2,
                     x_right_down_2, y_right_down_2):
    src = np.array([
        [x_left_top_1, y_left_top_1],
        [x_right_down_1, y_left_top_1],
        [x_left_top_1, y_right_down_1]], dtype=np.float32)

    dest = np.array([
        [x_left_top_2, y_left_top_2],
        [x_right_down_2, y_left_top_2],
        [x_left_top_2, y_right_down_2]], dtype=np.float32)

    M = cv2.getAffineTransform(src, dest)
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow('img', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


window_transform(0, 0, 487, 487,
                 12, 13, 140, 300)
