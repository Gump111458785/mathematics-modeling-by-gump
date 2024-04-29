import numpy as np
import cv2


def vif(image_ref, image_dist):
    # 将图像转换为灰度图像
    image_ref_gray = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
    image_dist_gray = cv2.cvtColor(image_dist, cv2.COLOR_BGR2GRAY)

    # 使用OpenCV的高斯滤波器进行模糊处理，模拟人眼对细节的感知
    image_ref_blur = cv2.GaussianBlur(image_ref_gray, (11, 11), 1.5)
    image_dist_blur = cv2.GaussianBlur(image_dist_gray, (11, 11), 1.5)

    # 计算结构信息（局部特征的标准差）
    ksize = 3
    image_ref_std = cv2.Sobel(image_ref_blur, cv2.CV_64F, 1, 1, ksize=ksize).std()
    image_dist_std = cv2.Sobel(image_dist_blur, cv2.CV_64F, 1, 1, ksize=ksize).std()

    # 计算失真感知性量化值（结构信息的均方差）
    distortion_sensitivity = (image_dist_std ** 2) / (image_ref_std ** 2)

    # 计算结构敏感性量化值（结构信息的相关性）
    cov_matrix = np.cov(image_ref_blur.flatten(), image_dist_blur.flatten())
    correlation = cov_matrix[0, 1] / (image_ref_std * image_dist_std)

    # 计算VIF分数
    vif_score = distortion_sensitivity * correlation

    return vif_score

# Example usage:
# img1 = np.random.rand(256, 256, 3)  # Example first image
# img2 = np.random.rand(256, 256, 3)  # Example second image
# vif_value = vifvec(img1, img2)
# print("VIF value:", vif_value)
