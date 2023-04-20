'''
图像的平滑滤波

1. 调用函数，例如matlab的filter,opencv的blur等函数，实现均值滤波，输出平滑后的图像，比较与分析不同参数下的实验效果
'''
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# task1
img = cv.imread("Figure/2.jpg")

img1 = cv.blur(img, (3, 3))
img2 = cv.blur(img, (5, 5))
img3 = cv.blur(img, (7, 7))

tmp1 = np.hstack((img, img1))
tmp2 = np.hstack((img2, img3))
tmp = np.vstack((tmp1, tmp2))

cv.imshow('image', tmp)
cv.waitKey(0)
cv.destroyAllWindows()