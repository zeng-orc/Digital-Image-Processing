'''
图像二值分割

1. 利用灰度直方图求双峰或多峰，选择两峰之间的谷底作为阈值，将图像分割为前景和背景部分，结果显示为二值图像。Matlab中的相关函数为im2bw(I,T)，I为待分割图像，T为分割阈值。
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 计算灰度直方图
def calcGrayHist(grayimage):
    # 灰度图像矩阵的高，宽
    rows, cols = grayimage.shape

    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[grayimage[r][c]] += 1

    return grayHist

# 阈值分割：直方图技术法
def threshTwoPeaks(image):

    #转换为灰度图
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 计算灰度直方图
    histogram = calcGrayHist(gray)
    # 寻找灰度直方图的最大峰值对应的灰度值
    maxLoc = np.where(histogram == np.max(histogram))
    # print(maxLoc)
    firstPeak = maxLoc[0][0] #灰度值
    # 寻找灰度直方图的第二个峰值对应的灰度值
    measureDists = np.zeros([256], np.float32)
    for k in range(256):
        measureDists[k] = pow(k - firstPeak, 2) * histogram[k] #综合考虑 两峰距离与峰值
    maxLoc2 = np.where(measureDists == np.max(measureDists))
    secondPeak = maxLoc2[0][0]
    print('双峰为：',firstPeak,secondPeak)

    # 找到两个峰值之间的最小值对应的灰度值，作为阈值
    thresh = 0
    if firstPeak > secondPeak:  # 第一个峰值再第二个峰值的右侧
        temp = histogram[int(secondPeak):int(firstPeak)]
        minloc = np.where(temp == np.min(temp))
        thresh = secondPeak + minloc[0][0] + 1
    else:  # 第一个峰值再第二个峰值的左侧
        temp = histogram[int(firstPeak):int(secondPeak)]
        minloc = np.where(temp == np.min(temp))
        thresh = firstPeak + minloc[0][0] + 1

    # 找到阈值之后进行阈值处理，得到二值图
    threshImage_out = gray.copy()
    # 大于阈值的都设置为255
    threshImage_out[threshImage_out > thresh] = 255
    threshImage_out[threshImage_out <= thresh] = 0
    return thresh, threshImage_out

if __name__ == "__main__":

    img = cv.cvtColor(cv.imread('Figure/2.jpg'), cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    #灰度直方图曲线
    hist = cv.calcHist([img_gray], [0], None, [256], [0, 255]) #对图像像素的统计分布，它统计了每个像素（0到L-1）的数量。

    thresh, img_sep = threshTwoPeaks(img)
    print('灰度阈值为:',thresh)

    plt.figure('imgs')
    plt.subplot(221).set_title('original_img')
    plt.imshow(img)
    plt.subplot(222).set_title('gray_img')
    plt.imshow(img_gray, cmap="gray")
    plt.subplot(223).set_title('hist')
    plt.hist(img_gray.ravel(), 256)
    plt.subplot(224).set_title('hist_sep')
    plt.imshow(img_sep, cmap="gray")
    plt.show()
