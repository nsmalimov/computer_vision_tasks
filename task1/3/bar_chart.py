import cv2

import numpy as np
from matplotlib import pyplot as plt


def f_transform(x):
    return 255 * (x - h_min) / (h_max - h_min + 0.0)


img = cv2.imread('low-contrast.png', 0)

hist, bins = np.histogram(img.ravel(), 256, [0, 256])

bins = bins[0:-1]

h_min = 0
h_max = 255

count = 0

for i in hist:
    if (i > 55):
        h_min = count
        break
    count += 1

last = 0
hits_1 = list(hist)
count = 0

for i in hist:
    if (hits_1.count(i) >= 30):
        if (last < i):
            last = i
            h_max = count
    count += 1

print h_min, h_max


def plot_hist_orig():
    ax = plt.subplot()
    ax.bar(bins, hist, width=0.2, color='b', align='center')

    plt.show()


def plot_hist():
    new_hist = []

    for i in hist:
        new_hist.append(f_transform(i))

    ax = plt.subplot()
    ax.bar(bins, new_hist, width=0.2, color='b', align='center')

    plt.show()


def up_contrast(img):
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if img[i, j] <= h_min:
                img[i, j] = h_min
            elif img[i, j] >= h_max:
                img[i, j] = h_max
            else:
                img[i, j] = f_transform(img[i, j])

    return img


up_contrast_img = up_contrast(img)

cv2.imwrite('up-contrast.png', up_contrast_img)
