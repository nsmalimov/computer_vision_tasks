#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import cv2
import numpy as np


def draw_and_save():
    img = np.ones((400, 400)) * 255
    cv2.circle(img, (90, 98), 5, (0, 0, 255), -1)
    cv2.circle(img, (145, 100), 5, (0, 0, 255), -1)
    cv2.circle(img, (70, 116), 5, (0, 0, 255), -1)
    cv2.circle(img, (250, 200), 5, (0, 0, 255), -1)
    cv2.imwrite("image.png", img)


def get_center_circle():
    img = cv2.imread('image.png')
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


def draw_one_line(vector, img):
    x = np.arange(0, img.shape[0])
    for i in x:
        y = ((-1) * vector[0] * i + (-1) * vector[2]) / vector[1]

        cv2.circle(img, (i, y), 1, (0, 0, 0), -1)


def draw_line():
    circle_middle = get_center_circle()

    img = cv2.imread('image.png')

    count = 0
    was_looked = []
    for i in circle_middle:
        for j in circle_middle:
            if (list([i, j]) in was_looked or list([j, i]) in was_looked):
                continue
            if (i != j):
                a = [i[0], i[1], 1]
                b = [j[0], j[1], 1]
                vector = np.cross(a, b)

                count += 1
                cv2.line(img, (i[0], i[1]), (j[0], j[1]), (0, 0, 0))
                draw_one_line(vector, img)
                was_looked.append([i, j])

    cv2.imshow('img', img)
    cv2.imwrite("image1.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_line_coef():
    circle_middle = get_center_circle()

    all_lines = []

    was_looked = []

    for i in circle_middle:
        for j in circle_middle:
            if (list([i, j]) in was_looked or list([j, i]) in was_looked):
                continue
            if (i != j):
                a = [i[0], i[1], 1]
                b = [j[0], j[1], 1]
                vector = np.cross(a, b)

                was_looked.append([i, j])
                all_lines.append(copy.deepcopy(vector))

    return all_lines


def check_equal_vector(first, second):
    for i in first:
        for j in second:
            if (i != j):
                return False
    return True


def draw_intersection():
    all_vectors = get_line_coef()

    img = cv2.imread('image1.png')

    for i in all_vectors:
        for j in all_vectors:
            if not (check_equal_vector(i, j)):
                vector = np.cross(i, j)

                print vector

                # получаем вектор в однородных координатах (T, U, V)
                # нормируем, (T/V, U/V, 1)
                homogeneous_vector = [vector[0] / vector[-1], vector[1] / vector[-1], 1]

                # получаем (T/V, U/V) — искомая точка на плоскости.
                found_dot = [homogeneous_vector[0], homogeneous_vector[1]]
                cv2.circle(img, (found_dot[0], found_dot[1]), 5, (0, 0, 255), -1)

    cv2.imshow('img', img)
    cv2.imwrite("image2.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# draw_and_save()
# draw_line()

draw_intersection()
