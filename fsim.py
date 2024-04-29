import cv2
import numpy as np

def fsim(image_ref, image_dist):
    # 转换图像为灰度
    image_ref_gray = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
    image_dist_gray = cv2.cvtColor(image_dist, cv2.COLOR_BGR2GRAY)

    # 计算FSIM
    mu_ref = np.mean(image_ref_gray)
    mu_dist = np.mean(image_dist_gray)
    sigma_ref = np.std(image_ref_gray)
    sigma_dist = np.std(image_dist_gray)
    sigma_ref_dist = np.mean((image_ref_gray - mu_ref) * (image_dist_gray - mu_dist))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    fsim_score = ((2 * mu_ref * mu_dist + c1) * (2 * sigma_ref_dist + c2)) / (
                (mu_ref ** 2 + mu_dist ** 2 + c1) * (sigma_ref ** 2 + sigma_dist ** 2 + c2))

    return fsim_score
#!/usr/bin/env python
# -*- coding:utf-8 -*-
