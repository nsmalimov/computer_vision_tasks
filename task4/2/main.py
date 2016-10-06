# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy.optimize import leastsq


def to_affine(x):
    x[0] = x[0] / x[2]
    x[1] = x[1] / x[2]
    return x[0:2]


def func(x):
    R = x[0:3]
    t = x[3:]

    t = np.transpose(t)
    rotation = cv2.Rodrigues(R)[0]

    res = []

    for k in xrange(len(plane_points[0:i])):
        res.append(np.dot(calibration_matrix,
                          np.dot(rotation, np.insert(plane_points[0:i][k], 2, np.float32(0.0), axis=0)) + t))

    for k in xrange(len(res)):
        res[k] = to_affine(res[k])

    # минимизация разницы
    ans = (res - image_points[0:i]).flatten()

    return ans


def write_in_file(filename, ans):
    f = open(filename, 'a')
    f.write(str(ans) + "\n")
    f.close()


def clarification_func(rotation_param):
    x0 = rotation_param  # стартовая точка минимизации
    rotation_param = leastsq(func, x0)[0]

    # считаем положение самой точки
    t = rotation_param[3:6]
    R = cv2.Rodrigues(rotation_param[0:3])[0]

    return R, t


def dot_predict():
    ans = []

    for j in xrange(len(plane_points)):
        # точка фактическое положение
        a = np.insert(plane_points[j], 2, np.float32(0), axis=0)  # добавить 0 в конец

        # переходим в систему координат камеры
        some = np.dot(R, a) + t  # P * ()

        # проектируем на камеру
        predicted_dot = np.dot(calibration_matrix, some)

        predicted_dot = to_affine(predicted_dot)

        ans.append(predicted_dot)

    return ans


img = cv2.imread('chessboard.png', 0)
filename = "answer.txt"

cell_len = 3.33
focus = 6741

image_points = np.float32(
    [[202, 160], [426, 540], [255, 435], [738, 615], [820, 307], [789, 170], [509, 233], [992, 412]])

plane_coord = np.float32([[100, 0], [200, 300], [100, 200], [400, 400], [500, 200], [500, 100], [300, 100], [600, 300]])
plane_points = cell_len * plane_coord

# находим матрицу гомографии по первым четырём точкам
hm_matrix = cv2.findHomography(plane_coord[0:4], image_points[0:4])

# матрица камеры
# калибровачные параметры камеры
calibration_matrix = np.float32([[focus, 0, img.shape[1] / 2], [0, focus, img.shape[0] / 2], [0, 0, 1]])

# обратная матрица
calibration_matrix_inversed = np.linalg.inv(calibration_matrix)

# print calibration_matrix

H_n = np.dot(calibration_matrix_inversed, hm_matrix[0])

# делим на норму первого столбца
H_n = H_n / np.linalg.norm(H_n[:, 0])

r1 = H_n[:, 0]  # поворот
r2 = H_n[:, 1]  # поворот
t0 = H_n[:, 2]  # вектор переноса

r3 = np.cross(r1, r2)  # векторное произведение

R = np.column_stack((r1, r2, r3))  # матрица поворота

rotation_param = cv2.Rodrigues(R)[0]  # пареметризация вращений
rotation_param_T = np.transpose(rotation_param)[0]

R_f = np.concatenate((rotation_param_T, t0), axis=0)

task_array = [4, 5, 6, 7, 8]

for i in task_array:

    # уточним положение камеры по выбранным точкам
    R, t = clarification_func(R_f)

    # предскажем положение точек
    predicted_dot_array = dot_predict()

    sum = 0.0

    for j in xrange(len(predicted_dot_array)):
        sum += np.linalg.norm(predicted_dot_array[j] - image_points[j])

    sum = sum / 8

    write_in_file(filename, sum)
