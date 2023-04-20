'''
图像锐化

1. 自己编写程序，基于如下两种滤波核K1=|Gx|+|Gy|和K2=|G’x|+|G’y|进行图像锐化，输出结果图像，比较与分析两种滤波的实验效果
2. 自己设计一个3*3的模板（3*3），能够检测正负45度倾斜方向的图像细节
'''
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# task1
img = cv.imread("Figure/1.jpg", 0)

Gx1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
Gy1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=int)
Gx2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
Gy2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)

img1x = cv.filter2D(img, -1 ,Gx1)
img1y = cv.filter2D(img, -1 ,Gy1)
img2x = cv.filter2D(img, -1 ,Gx2)
img2y = cv.filter2D(img, -1 ,Gy2)

img1 = abs(img1x) + abs(img1y)
img2 = abs(img2x) + abs(img2y)

tmp = np.hstack((img, img1, img2))

cv.imshow('image', tmp)
cv.waitKey(0)
cv.destroyAllWindows()

# task2
Gx = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=int)
Gy = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=int)

img3x = cv.filter2D(img, -1, Gx)
img3y = cv.filter2D(img, -1, Gy)

img3 = abs(img3x) + abs(img3y)

tmp = np.hstack((img, img3))

cv.imshow('image', tmp)
cv.waitKey(0)
cv.destroyAllWindows()