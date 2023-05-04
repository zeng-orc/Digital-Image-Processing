'''
灰度图像的离散余弦变换

1. 对输入的灰度图像分别进行8*8（像素）和16*16的分块，对每块图像进行离散余弦变换，输出频谱图（DCT系数）
2. 自行修改部分的DCT系数
3. 通过离散余弦逆变换，显示还原与拼接后的图像块
'''
from matplotlib import pyplot as plt
import numpy as np
import cv2

from Work1 import set_ch

# 对输入的灰度图像分别进行8*8（像素）和16*16的分块，对每块图像进行离散余弦变换，输出频谱图（DCT系数）
def dct(img, num):
    rows, cols = img.shape
    rows, cols = int(rows / num) * num, int(cols / num) * num
    img = img[:rows, :cols]
    himg = np.vsplit(img, rows / num)
    dct_spectral_img = []
    for i in range(0, rows // num):
        blockimg = np.hsplit(himg[i], cols / 8)
        dct_spectral_himg = []
        for j in range(0, cols // 8):
            block = blockimg[j]
            dct_spectral_himg.append(cv2.dct(block.astype(float)))
        dct_spectral_img.append(dct_spectral_himg)
    return np.array(dct_spectral_img)

def block2img(block):
    rows, cols = block.shape[:2]
    img = np.array([])
    for i in range(0, rows):
        img_tmp = np.array([])
        for j in range (0, cols):
            if j == 0:
                img_tmp = block[i][j]
                continue
            img_tmp = np.hstack((img_tmp, block[i][j]))
        if i == 0:
            img = img_tmp
            continue
        img = np.vstack((img, img_tmp))
    return img

def show_spectral_img(img, dct_spectral_img):
    plt.subplot(1, 2, 1), plt.imshow(img, 'gray'), plt.title('原始图像')
    plt.subplot(1, 2, 2), plt.imshow(dct_spectral_img, 'gray'), plt.title('余弦频谱')
    plt.show()

# 通过离散余弦逆变换，显示还原与拼接后的图像块
def show_space_img(dct_spectral_block):
    dct_space_block = []
    for i in dct_spectral_block:
        dct_space_hblock = []
        for j in i:
            dct_space_hblock.append(cv2.idct(j))
        dct_space_block.append(dct_space_hblock)
    dct_space_img = block2img(np.array(dct_space_block))
    plt.subplot(1, 2, 1), plt.imshow(img, 'gray'), plt.title('原始图像')
    plt.subplot(1, 2, 2), plt.imshow(dct_space_img, 'gray'), plt.title('余弦图像')
    plt.show()

if __name__ == "__main__":
    set_ch()
    img = cv2.imread("Figure/2.jpg", 0)
    num = 16
    dct_spectral_block = dct(img, num)
    dct_spectral_img = block2img(dct_spectral_block)
    show_spectral_img(img, dct_spectral_img)
    show_space_img(dct_spectral_block)