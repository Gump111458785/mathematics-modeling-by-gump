import cv2
import numpy as np


def compute_wsnr(image_ref, image_dist):
    # 计算原始图像和失真图像的傅立叶变换
    ref_fft = np.fft.fft2(image_ref)
    dist_fft = np.fft.fft2(image_dist)

    # 计算傅立叶变换后的幅度谱
    ref_mag = np.abs(ref_fft)
    dist_mag = np.abs(dist_fft)

    # 计算信号和噪声
    signal = np.sum(ref_mag ** 2)
    noise = np.sum((ref_mag - dist_mag) ** 2)

    # 计算WSNR
    wsnr = 10 * np.log10(signal / noise)

    return wsnr
