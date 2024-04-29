import cv2
import numpy as np

def psnr_hma(img1, img2):
    # 计算均方误差
    mse = np.mean((img1 - img2) ** 2)
    # 计算 PSNR-HMA
    psnr_hma = 10 * np.log10(255**2 / mse)
    return psnr_hma