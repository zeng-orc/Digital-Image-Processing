'''
灰度图像的频域滤波器

1. 灰度图像进行离散傅里叶变换
2. 分别使用梯形的高通和低通滤波器进行滤波，显示滤波后的频谱图像
3. 显示离散傅里叶逆变换后的空域图像，观察振铃现象
'''
from matplotlib import pyplot as plt
import numpy as np
import cv2

# 中文显示工具函数
def set_ch():
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    mpl.rcParams['axes.unicode_minus'] = False

# 灰度图像进行离散傅里叶变换
def dis_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift))
    return fshift, fimg

# 分别使用梯形的高通和低通滤波器进行滤波，显示滤波后的频谱图像
def low_pass(img, fshift, D1, D2):
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols), dtype='float')
    for i in range(rows):
        for j in range(cols):
            dis = np.sqrt((crow - i) * (crow - i) + (ccol - j) * (ccol - j))
            if dis <= D1:
                mask[i, j] = 1
            elif dis <= D2:
                mask[i, j] = (D2 - dis) / (D2 - D1)
    fshift = fshift * mask
    return fshift

def  high_pass(img, fshift, D1, D2):
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            dis = np.sqrt((crow - i) * (crow - i) + (ccol - j) * (ccol - j))
            if dis >= D2:
                mask[i, j] = 1
            elif dis >= D1:
                mask[i, j] = (dis - D1) / (D2 - D1)
    fshift = fshift * mask
    return fshift

def show_spectral_img(img, fshift, fimg, D1=50, D2=100):
    fshift_low = low_pass(img, fshift, D1, D2)
    fshift_high = high_pass(img, fshift, D1, D2)

    fimg_low = np.log(np.abs(fshift_low))
    fimg_high = np.log(np.abs(fshift_high))

    plt.figure()
    plt.subplot(131)
    plt.imshow(fimg, cmap='gray')
    plt.title('傅里叶频谱')

    plt.subplot(132)
    plt.imshow(fimg_low, cmap='gray')
    plt.title('低通滤波后的频谱图像')

    plt.subplot(133)
    plt.imshow(fimg_high, cmap='gray')
    plt.title('高通滤波后的频谱图像')
    plt.show()
    return fshift_low, fshift_high

# 显示离散傅里叶逆变换后的空域图像，观察振铃现象
def show_space_img(img, fshift_low, fshift_high):
    fishift_low = np.fft.ifftshift(fshift_low)
    img_back_low = np.fft.ifft2(fishift_low)
    img_back_low = np.abs(img_back_low)
    img_back_low = (img_back_low - np.amin(img_back_low)) / (np.amax(img_back_low) - np.amin(img_back_low))

    fishift_high = np.fft.ifftshift(fshift_high)
    img_back_high = np.fft.ifft2(fishift_high)
    img_back_high = np.abs(img_back_high)
    img_back_high = (img_back_high - np.amin(img_back_high)) / (np.amax(img_back_high) - np.amin(img_back_high))

    plt.figure()
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('原始图像')

    plt.subplot(132)
    plt.imshow(img_back_low, cmap='gray')
    plt.title('低通滤波后的图像')

    plt.subplot(133)
    plt.imshow(img_back_high, cmap='gray')
    plt.title('高通滤波后的图像')
    plt.show()


if __name__ == "__main__":
    set_ch()
    img = cv2.imread("Figure/2.jpg", 0)
    fshift, fimg = dis_fft(img)
    plt.subplot(1, 2, 1), plt.imshow(img, 'gray'), plt.title('原始图像')
    plt.subplot(1, 2, 2), plt.imshow(fimg, 'gray'), plt.title('傅里叶频谱')
    plt.show()
    fshift_low, fshift_high = show_spectral_img(img, fshift, fimg)
    show_space_img(img, fshift_low, fshift_high)
