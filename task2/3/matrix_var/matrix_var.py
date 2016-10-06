#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import cv2
import numpy as np


def draw_square(dots_array, name_image):
    img = np.ones((400, 400)) * 255

    center = 200

    for i in xrange(len(dots_array)):
        for j in xrange(len(dots_array[i])):
            dots_array[i][j] = dots_array[i][j] + center

    dots = [dots_array[0], dots_array[1], dots_array[2], dots_array[3]]

    cv2.line(img, (dots[0][0], dots[0][1]), (dots[1][0], dots[1][1]), (0, 0, 0))

    cv2.line(img, (dots[1][0], dots[1][1]), (dots[2][0], dots[2][1]), (0, 0, 0))

    cv2.line(img, (dots[2][0], dots[2][1]), (dots[3][0], dots[3][1]), (0, 0, 0))

    cv2.line(img, (dots[3][0], dots[3][1]), (dots[0][0], dots[0][1]), (0, 0, 0))

    cv2.imwrite(name_image + ".png", img)


def ortographic(camera_dots):
    n_x = 400
    n_y = 400

    M_vp = [[n_x / (2 + 0.0), 0, 0, (n_x - 1) / (2 + 0.0)],
            [0, n_y / (2 + 0.0), 0, (n_y - 1) / (2 + 0.0)],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]

    M_vp = np.array(M_vp)

    # top and bottom of cube
    l = -200
    b = -200
    n = -20

    r = 200
    t = 200
    f = -350

    M_ort = [[2 / ((r - l) + 0.0), 0, 0, (-r + l) / ((r - l) + 0.0)],
             [0, 2 / ((t - b) + 0.0), 0, (-t + b) / ((t - b) + 0.0)],
             [0, 0, 2 / (n - f), (-n + f) / ((n - f) + 0.0)],
             [0, 0, 0, 1]]

    M_ort = np.matrix(M_ort)

    answer_dots = []

    for i in camera_dots:
        some_array = copy.deepcopy(i)
        some_array.append(1)

        camera_matrix = np.matrix(copy.deepcopy(some_array))

        answer_matrix = (M_vp * M_ort) * camera_matrix.T

        # print answer_matrix

        answer_dots.append(copy.deepcopy(answer_matrix[0:2]))

    # draw square
    draw_square(answer_dots, "image_ort1")


def perspective(camera_dots):
    n_x = 400
    n_y = 400

    M_vp = [[n_x / (2 + 0.0), 0, 0, (n_x - 1) / (2 + 0.0)],
            [0, n_y / (2 + 0.0), 0, (n_y - 1) / (2 + 0.0)],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]

    M_vp = np.array(M_vp)

    # top and bottom of cube
    l = -200
    b = -200
    n = -20

    r = 200
    t = 200
    f = -350

    p = np.array([[n, 0, 0, 0],
                  [0, n, 0, 0],
                  [0, 0, n + f, -f * n],
                  [0, 0, 1, 0]])

    M_ort = [[2 / ((r - l) + 0.0), 0, 0, (-r + l) / ((r - l) + 0.0)],
             [0, 2 / ((t - b) + 0.0), 0, (-t + b) / ((t - b) + 0.0)],
             [0, 0, 2 / (n - f), (-n + f) / ((n - f) + 0.0)],
             [0, 0, 0, 1]]

    M_ort = np.matrix(M_ort)

    answer_dots = []

    for i in camera_dots:
        some_array = [n * i[0] / (i[-1] + 0.0), n * i[1] / (i[-1] + 0.0), n + f - f * n / (i[-1] + 0.0), 1]

        camera_matrix = np.matrix(copy.deepcopy(some_array))
        answer_matrix = (M_vp * M_ort) * p * camera_matrix.T

        # print answer_matrix

        answer_dots.append(copy.deepcopy(answer_matrix[0:2]))

    # draw square
    draw_square(answer_dots, "image_persp1")


# camera(0,0,0)

square_dots = [[-80, 80, -113.13],
               [80, 80, -113.13],
               [80, -80, 113.13],
               [-80, -80, 113.13]]

ortographic(square_dots)

perspective(square_dots)
