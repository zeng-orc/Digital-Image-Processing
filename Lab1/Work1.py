'''
图像的对比度变换

1. 提取彩色图像的亮度通道进行对比度调整，自行设计至少一种变换函数，例如伽马变换、分段线性变换等
2. 首先将彩色图像转换为灰度图像，使用问题1中的同一个函数进行变换
3. 比较处理后的亮度图像和处理后的灰度图像，分析发生的效果变化
'''
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def Gamma(r, c = 1, gamma = 1):
    # gamma值小于1时，会拉伸图像中灰度级较低的区域，同时会压缩灰度级较高的部分，与此对应的变化是图像的暗部细节会得到提升，建议>0.7
    # gamma值大于1时，会拉伸图像中灰度级较高的区域，同时会压缩灰度级较低的部分，这样处理和的结果是图像的对比度得到明显提升，建议<1.4
    return c * np.power(r, gamma)

# task1
img = cv.imread("Figure/1.jpg")
img_t = cv.cvtColor(img,cv.COLOR_BGR2HSV)
h,s,v = cv.split(img_t)

v1 = np.clip(Gamma(v, gamma=1.4),0,255)

img1 = np.uint8(cv.merge((h,s,np.uint8(v1))))
img1 = cv.cvtColor(img1,cv.COLOR_HSV2BGR)

tmp = np.hstack((img, img1))

cv.imshow('image', tmp)
cv.waitKey(0)
cv.destroyAllWindows()

# task2
img_t = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
g = cv.split(img_t)

g2 = np.clip(Gamma(g, gamma=1.4), 0, 255)

img2 = np.uint8(cv.merge((np.uint8(g2))))

tmp = np.hstack((img_t, img2))

cv.imshow('image', tmp)
cv.waitKey(0)
cv.destroyAllWindows()