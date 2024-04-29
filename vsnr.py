import cv2
import numpy as np

def vsnr(image_ref, image_dist):
    # 转换图像为灰度
    image_ref_gray = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
    image_dist_gray = cv2.cvtColor(image_dist, cv2.COLOR_BGR2GRAY)

    # 计算信号
    signal = np.mean(image_ref_gray)

    # 计算噪声
    noise = np.mean((image_ref_gray - image_dist_gray) ** 2)

    # 计算VSNR
    vsnr_score = 20 * np.log10(signal / np.sqrt(noise))

    return vsnr_score
