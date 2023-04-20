'''
Hough线检测

1. 对作业一中边缘检测的结果，进行Hough线检测
2. 测试多组（不少于3组）参数，提取较长的边界，分析结果
'''

import numpy as np
import cv2

from Work1 import x_roberts, y_roberts, x_prewitt, y_prewitt, robert_processing, prewitt_processing


def img_processing(img):
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # robert
    # x_edge_img, y_edge_img, edges = robert_processing(gray)
    # prewitt
    edge_x_img, edge_y_img, edge_img_max, edge_img_sum, edge_img_abs, edges = prewitt_processing(gray)
    return np.array(edges, np.uint8)


def line_detect(img):
    img = cv2.imread(img)
    result = img_processing(img)  # 返回来的是一个矩阵
    # 霍夫线检测
    # 统计概率霍夫线变换函数：图像矩阵，极坐标两个参数，一条直线所需最少的曲线交点，组成一条直线的最少点的数量，被认为在一条直线上的亮点的最大距离
    lines = cv2.HoughLinesP(result, 1, 1 * np.pi / 180, 10, minLineLength=50, maxLineGap=5)
    print("Line Num : ", len(lines))

    # 画出检测的线段
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        pass

    # np.uint8(cv2.merge((np.uint8(img))))
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    line_detect("Figure/1.jpg")
    pass
