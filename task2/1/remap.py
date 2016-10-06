#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from scipy.interpolate import griddata

img = cv2.imread('image.jpg')
rows, cols = img.shape[:2]


def window_transform(x_left_top_1, y_left_top_1, x_right_down_1, y_right_down_1, x_left_top_2, y_left_top_2,
                     x_right_down_2, y_right_down_2):
    grid_x, grid_y = np.mgrid[0:rows, 0:cols]

    destination = np.array([

        [y_right_down_2, y_left_top_2],

        [x_left_top_2, y_left_top_2],

        [x_left_top_2, x_right_down_2],

        [y_right_down_2, x_right_down_2]])

    source = np.array([
        [rows, 0],
        [0, 0],
        [0, cols],
        [rows, cols]])

    grid_z = griddata(destination, source, (grid_x, grid_y), method='linear')
    map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(rows, cols)
    map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(rows, cols)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    dst = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_LINEAR)

    cv2.imshow('img', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


window_transform(0, 0, 487, 487,
                 12, 13, 140, 300)
