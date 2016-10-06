#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import cv2
import numpy as np


def get_center_circle(filename):
    img = cv2.imread(filename)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centres = []
    for i in range(len(contours)):
        moments = cv2.moments(contours[i])
        centres.append((int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])))
        cv2.circle(img, centres[-1], 3, (0, 0, 0), -1)

    new_array = []
    for i in centres:
        new_array.append(list(i))

    return new_array[1:]


def create_new_dots():
    new_points = np.float32([[579, 119, 1], [580, 235, 1], [543, 200, 1], [584, 208, 1]])

    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            img[i][j] = [255, 255, 255]

    for x, y, c in new_points:
        img[y, x] = [0, 0, 0]  # mark by green

    cv2.imwrite('marked_image.png', img)

    image = cv2.warpPerspective(img, hom, (400, 400))

    cv2.imwrite('marked_expand.png', image)

    return new_points, image


def find_nearest_dots(black_dots, x_predict, y_predict):
    e = 1000

    x_select = 0
    y_select = 0

    for i in black_dots:
        diff = math.sqrt((i[0] - x_predict) ** 2 + (i[1] - y_predict) ** 2)

        if (diff < e):
            e = diff
            x_select = i[0]
            y_select = i[1]

    return x_select, y_select


def find_error():
    errors_array = []

    black_dots = get_center_circle("task3/marked_expand.png")

    for i in new_points:
        x, y, z0 = np.dot(hom, i)
        x_predict, y_predict = int(x / z0), int(y / z0)

        mark_x, mark_y = find_nearest_dots(black_dots, x_predict, y_predict)

        errors_array.append(math.sqrt((x_predict - mark_x) ** 2 + (y_predict - mark_y) ** 2))

    return errors_array


coord_in_image = get_center_circle('task3/hall405_prepared.png')

coord_in_image = np.float32(coord_in_image)

coord_in_object = np.float32([[0, 399], [399, 399], [0, 0], [399, 0]])

hom = cv2.findHomography(coord_in_image, coord_in_object, cv2.RANSAC, 5.0)[0]

img = cv2.imread('task3/hall405.jpg')
rows, cols = img.shape[:2]

image = cv2.warpPerspective(img, hom, (400, 400))

# развернутое изображение в снимок виртуальной камеры
cv2.imwrite('picture_expand.png', image)

# выбрали новые точки и отметили на изображениях
new_points, marked_image = create_new_dots()

errors_array = find_error()

print errors_array
