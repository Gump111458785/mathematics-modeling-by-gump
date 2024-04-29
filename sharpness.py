import cv2


def compute_sharpness(image):
    # 使用拉普拉斯算子计算图像的梯度
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # 计算梯度的绝对值
    abs_laplacian = cv2.convertScaleAbs(laplacian)

    # 计算梯度的平均值作为锐度评分
    sharpness_score = cv2.mean(abs_laplacian)[0]

    return sharpness_score
def sharpness(img1,img2):
    return compute_sharpness(img2)/compute_sharpness(img1)
#!/usr/bin/env python
# -*- coding:utf-8 -*-
