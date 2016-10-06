# -*- coding: utf-8 -*-

import math
import random

import cv2
import numpy as np
from scipy.optimize import leastsq


def to_affine(x):
    temp = 0.0001
    if (x[2] != 0):
        temp = x[2]

    x[0] = x[0] / temp
    x[1] = x[1] / temp
    return x[0:2]


# производим уточнение
def clarification_func(h):
    calibration_matrix_inversed = np.linalg.inv(calibration_matrix)
    H_n = np.dot(calibration_matrix_inversed, h)

    # делим на норму первого столбца
    Hn = H_n / np.linalg.norm(H_n[:, 0])

    r1 = Hn[:, 0]  # поворот
    r2 = Hn[:, 1]  # поворот
    t0 = Hn[:, 2]  # вектор переноса
    r3 = np.cross(r1, r2)  # векторное произведение
    R = np.column_stack((r1, r2, r3))  # матрица поворота

    R_f = cv2.Rodrigues(R)[0]  # параметризация вращений
    R_f = np.transpose(R_f)[0]
    R_f = np.concatenate((R_f, t0), 0)

    # находим наилучшую параметризацию вращений
    R_f_best = leastsq(func, R_f)[0][0:3]

    return cv2.Rodrigues(R_f_best)[0]


def func(x):
    R = x[0:3]
    t = x[3:]

    t = np.transpose(t)  # вектор переноса
    rotation = cv2.Rodrigues(R)[0]  # параметризация вращений

    res = []

    for k in plane_points[selected_dot]:
        k_clone = k
        k_clone = np.insert(k_clone, 2, np.float32(0.0), axis=0)

        res.append(np.dot(calibration_matrix, np.dot(rotation, k_clone + t)))

    for k in xrange(len(res)):
        res[k] = to_affine(res[k])

    return (res - image_points[selected_dot]).flatten()


def ransiac():
    selected_dot = []
    iter_dots = []
    h_new = []

    # количество итераций
    for i in range(inter_count):

        # выбираем случайные 4 точки
        c = random.sample(range(0, 16), 4)

        # выдвигаем гипотезу на основе этих точек
        # находим гомографию по выбранным точкам
        h = cv2.findHomography(plane_points[c], image_points[c])[0]

        if h is not None:

            # предсказываем положение точек по матрице гомграфии
            points_h = map(lambda x: to_affine(np.dot(h, np.insert(x, 2, 1, axis=0))), plane_points)
            diff = image_points - points_h

            # проверяем соответствие каждой точки модели

            check_points = []

            for i in diff:
                ans = i[0] ** 2 + i[1] ** 2
                if (ans < 20):
                    check_points.append(i)

            # если точек прошло больше, чем в прошлый раз
            if len(check_points) > len(iter_dots):
                # сколько прошло в этот раз
                iter_dots = np.copy(check_points)

                # записываем какие выбраны
                selected_dot = np.copy(c)
                h_new = h

    return selected_dot, h_new


img = cv2.imread('task4/3/chessboard.png', 0)
cell_len = 3.33

# https://en.wikipedia.org/wiki/RANSAC#Parameters
# веротяность, что на 4 точках будут выбраны только инлаеры
p = 0.99
n = 4

w = 0.5  # веротяность выбора хотя бы одного инлаера (выброса) на всех точках

# предпологаемое количество итераций

inter_count = math.log(1 - p) / math.log(1 - math.pow(w, n))
inter_count = int(inter_count)

image_points = np.float32(
    [[202, 160], [426, 540], [255, 435], [738, 615], [820, 307], [789, 170], [509, 233], [992, 412]])
plane_coord = np.float32([[100, 0], [200, 300], [100, 200], [400, 400], [500, 200], [500, 100], [300, 100], [600, 300]])
plane_points = cell_len * plane_coord

# добавляем случайные 8 точек
for i in range(8):
    image_points = np.vstack(
        (image_points, np.float32([random.randint(1, img.shape[1]), random.randint(1, img.shape[0])])))
    plane_points = np.vstack(
        (plane_points, np.float32([cell_len * random.randint(0, 5), cell_len * random.randint(0, 5)])))

# переопределение для оптимизации
calibration_matrix = np.float32([[6741, 0, img.shape[1] / 2], [0, 6741, img.shape[0] / 2], [0, 0, 1]])

# выбор точек
selected_dot, h = ransiac()

print "точки которые подходят: " + str(selected_dot)

# уточнение по выбранным точкам
R = clarification_func(h)

print "Уточненное положение камеры: " + str(R)
