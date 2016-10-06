#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import cv2
import numpy as np


def draw_square(dots_array, name_image):
    img = np.ones((400, 400)) * 255

    cv2.line(img, dots_array[0], dots_array[1], [0, 0, 0])
    cv2.line(img, dots_array[1], dots_array[3], [0, 0, 0])
    cv2.line(img, dots_array[3], dots_array[2], [0, 0, 0])
    cv2.line(img, dots_array[2], dots_array[0], [0, 0, 0])

    cv2.imwrite(name_image + ".png", img)


def find_ortographic_dot(points):
    x = points[0]
    y = points[1]
    z = points[2]

    x_new = int(x * 400 / 4 + 400 / 2)
    y_new = int(y * 400 / 4 + 400 / 2)

    return x_new, y_new


def ortographic(camera_dots):
    new_points = []

    for i in camera_dots:
        new_points.append(find_ortographic_dot(i))

    draw_square(new_points, "image_ort")


def find_perspective_dot(points, c_x, angle):
    a_x = points[0]
    a_y = points[1]
    a_z = points[2]

    f = c_x / (math.tan(math.radians(angle)) + 0.0)

    dot_x = int(f * (a_x / (a_z + 0.0)) + c_x)
    dot_y = int(f * (a_y / (a_z + 0.0)) + c_x)

    return dot_x, dot_y


def perspective(camera_dots):
    new_points = []

    for i in camera_dots:
        new_points.append(find_perspective_dot(i, 200, 45))

    draw_square(new_points, "image_persp")


# camera(0,0,0)
square_dots = [(0.7, 0.7, 1),
               (-0.7, 0.7, 1),

               (0.7, -0.7, 2),
               (-0.7, -0.7, 2)]

ortographic(square_dots)

perspective(square_dots)
