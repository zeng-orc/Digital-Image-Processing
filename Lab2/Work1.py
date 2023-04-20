'''
使用一阶检测算子（导数）进行图像的边缘检测

1. 自行编写程序而非调用函数，使用如下算子中的任何二个，进行图像的边缘提取。分析和比较两个算子的不同效果。
'''

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

x_roberts = np.array([[1, 0],
                      [0, -1]])
y_roberts = np.array([[0, 1],
                      [-1, 0]])
x_prewitt = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0, -1]])
y_prewitt = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [-1, -1, -1]])


def robert_cal(img, filter):
    h, w = img.shape
    img_filter = np.zeros([h, w])
    for i in range(h-1):
        for j in range(w-1):
            img_filter[i][j] = img[i][j]*filter[0][0]+img[i][j+1] * \
                filter[0][1]+img[i+1][j]*filter[1][0] + \
                img[i+1][j+1]*filter[1][1]
    return img_filter


def conv_calculate(img, filter):
    h, w = img.shape
    conv_img = np.zeros([h-2, w-2])
    for i in range(h-2):
        for j in range(w-2):
            conv_img[i][j] = img[i][j]*filter[0][0]+img[i][j+1]*filter[0][1]+img[i][j+2]*filter[0][2] +\
                img[i+1][j]*filter[1][0]+img[i+1][j+1]*filter[1][1]+img[i+1][j+2]*filter[1][2] +\
                img[i+2][j]*filter[2][0]+img[i+2][j+1] * \
                filter[2][1]+img[i+2][j+2]*filter[2][2]

    return conv_img


def robert_processing(gray_img):
    h, w = gray_img.shape
    img = np.zeros([h+1, w+1])
    img[1:h+1, 1:w+1] = gray_img[0:h, 0:w]

    x_edge_img = robert_cal(img, x_roberts)
    y_edge_img = robert_cal(img, y_roberts)
    edge_img = np.zeros([h, w])
    for i in range(h):
        for j in range(w):
            edge_img[i][j] = np.sqrt(
                x_edge_img[i][j]**2+y_edge_img[i][j]**2)/(np.sqrt(2))

    return x_edge_img, y_edge_img, edge_img


def prewitt_processing(gray_img):
    h, w = gray_img.shape
    img = np.zeros([h+2, w+2])
    img[2:h+2, 2:w+2] = gray_img[0:h]
    edge_x_img = conv_calculate(img, x_prewitt)
    edge_y_img = conv_calculate(img, y_prewitt)

    # p(i,j)=max[edge_x_img,edge_y_img]
    edge_img_max = np.zeros([h, w])
    for i in range(h):
        for j in range(w):
            if edge_x_img[i][j] > edge_y_img[i][j]:
                edge_img_max = edge_x_img[i][j]
            else:
                edge_img_max = edge_y_img

    # p(i,j)=edge_x_img+edge_y_img
    edge_img_sum = np.zeros([h, w])
    for i in range(h):
        for j in range(w):
            edge_img_sum[i][j] = edge_x_img[i][j]+edge_y_img[i][j]

    # p(i,j)=|edge_x_img|+|edge_y_img|
    edge_img_abs = np.zeros([h, w])
    for i in range(h):
        for j in range(w):
            edge_img_abs[i][j] = abs(edge_x_img[i][j]) + abs(edge_y_img[i][j])

    # p(i,j)=sqrt(edge_x_img**2+edge_y_img**2)
    edge_img_sqrt = np.zeros([h, w])
    for i in range(h):
        for j in range(w):
            edge_img_sqrt[i][j] = np.sqrt(
                (edge_x_img[i][j])**2+(edge_y_img[i][j])**2)

    return edge_x_img, edge_y_img, edge_img_max, edge_img_sum, edge_img_abs, edge_img_sqrt

if __name__ == "__main__":
    original_img = cv.imread("Figure/1.jpg")
    gray_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    original_img = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)

    # robert算子
    x_edge_img, y_edge_img, edge_img = robert_processing(gray_img)
    plt.figure('imgs_robert')
    plt.subplot(321).set_title('original_img')
    plt.imshow(original_img)
    plt.subplot(322).set_title('gray_img')
    plt.imshow(gray_img, cmap="gray")
    plt.subplot(323).set_title('x_edge_img')
    plt.imshow(x_edge_img, cmap="gray")
    plt.subplot(324).set_title('y_edge_img')
    plt.imshow(y_edge_img, cmap="gray")
    plt.subplot(325).set_title('edge_img')
    plt.imshow(edge_img, cmap="gray")
    plt.show()

    # prewitt算子
    edge_x_img, edge_y_img, edge_img_max, edge_img_sum, edge_img_abs, edge_img_sqrt = prewitt_processing(gray_img)
    plt.figure('imgs_prewitt')
    plt.subplot(331).set_title('original_img')
    plt.imshow(original_img)
    plt.subplot(332).set_title('gray_img')
    plt.imshow(gray_img, cmap="gray")
    plt.subplot(333).set_title('x_edge_img')
    plt.imshow(edge_x_img, cmap="gray")
    plt.subplot(334).set_title('y_edge_img')
    plt.imshow(edge_y_img, cmap="gray")
    plt.subplot(335).set_title('edge_img_max')
    plt.imshow(edge_img_max, cmap="gray")
    plt.subplot(336).set_title('edge_img_sum')
    plt.imshow(edge_img_sum, cmap="gray")
    plt.subplot(337).set_title('edge_img_sqrt')
    plt.imshow(edge_img_sqrt, cmap="gray")
    plt.subplot(338).set_title('edge_img_abs')
    plt.imshow(edge_img_abs, cmap="gray")
    plt.show()
    pass
