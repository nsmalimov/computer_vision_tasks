# -*- coding: utf-8 -*-
import math

import numpy as np
import numpy.linalg as linalg
import numpy.random as random


# Ax+By+Cz+D=0

# D = 0 - через начало координат

def normalize(v):
    norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    ans = v / norm
    return ans


def gen_dot(plane_basis):
    ans = []
    for i in xrange(100):
        ans.append(random.uniform() * plane_basis[0] + random.uniform() * plane_basis[1])
    return ans


def noise(dots):
    for i in xrange(len(dots)):
        dots[i] = dots[i] + random.normal(0, 0.1)
    return dots


def find_plane_func(num, dots_array):
    dots_array = dots_array[0:num]

    x = []
    y = []
    z = []

    for i in dots_array:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    ones = np.ones(len(dots_array))  # еденичная матрица

    A = np.column_stack((x, y, ones))

    A_T = A.copy().T

    z = z.copy().reshape(len(dots_array), 1)

    # x = (A^T*A)^(-1)*A^T*b
    res = linalg.inv(A_T.dot(A)).dot(A_T).dot(z)

    a = res[0][0]
    b = res[1][0]
    c = -1

    return normalize(np.array([a, b, c]))


def diff(plane):
    vector = np.cross(plane_basis[0], plane_basis[1])  # вектор нормали
    ans = linalg.norm(plane + normalize(vector))
    return ans


def write_in_file(filename, ans):
    f = open(filename, 'a')
    f.write(str(ans) + "\n")
    f.close()


task_array = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

plane_basis = [np.array([1, 0.5, 1]), np.array([-1, 1, 0.5])]

dots = gen_dot(plane_basis)

dots = noise(dots)

filename = "answer.txt"

for i in task_array:
    found_plane = find_plane_func(i, dots)
    diff_plane = diff(found_plane)
    write_in_file(filename, diff_plane)
