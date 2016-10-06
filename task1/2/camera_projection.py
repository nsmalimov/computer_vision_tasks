import math


def main_func(a_x, a_y, a_z):
    f = 600.0 * (math.sqrt(3) / (2 + 0.0))

    A_x = f * a_x / (a_z + 0.0) + 300
    A_y = f * a_y / (a_z + 0.0) + 300

    return A_x, A_y


A_x, A_y = main_func(-1, 1, 6)

print A_x, A_y
