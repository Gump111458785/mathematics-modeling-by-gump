import cv2
import numpy as np


def fsim_color(image_ref, image_dist):
    # 计算亮度分量
    image_ref_YCrCb = cv2.cvtColor(image_ref, cv2.COLOR_BGR2YCrCb)
    image_dist_YCrCb = cv2.cvtColor(image_dist, cv2.COLOR_BGR2YCrCb)
    Y_ref = image_ref_YCrCb[:, :, 0]
    Y_dist = image_dist_YCrCb[:, :, 0]

    # 计算颜色分量
    Cb_ref = image_ref_YCrCb[:, :, 1]
    Cb_dist = image_dist_YCrCb[:, :, 1]
    Cr_ref = image_ref_YCrCb[:, :, 2]
    Cr_dist = image_dist_YCrCb[:, :, 2]

    # 计算FSIM
    mu_Y_ref = np.mean(Y_ref)
    mu_Y_dist = np.mean(Y_dist)
    sigma_Y_ref = np.std(Y_ref)
    sigma_Y_dist = np.std(Y_dist)
    sigma_Y_ref_dist = np.mean((Y_ref - mu_Y_ref) * (Y_dist - mu_Y_dist))

    mu_Cb_ref = np.mean(Cb_ref)
    mu_Cb_dist = np.mean(Cb_dist)
    sigma_Cb_ref = np.std(Cb_ref)
    sigma_Cb_dist = np.std(Cb_dist)
    sigma_Cb_ref_dist = np.mean((Cb_ref - mu_Cb_ref) * (Cb_dist - mu_Cb_dist))

    mu_Cr_ref = np.mean(Cr_ref)
    mu_Cr_dist = np.mean(Cr_dist)
    sigma_Cr_ref = np.std(Cr_ref)
    sigma_Cr_dist = np.std(Cr_dist)
    sigma_Cr_ref_dist = np.mean((Cr_ref - mu_Cr_ref) * (Cr_dist - mu_Cr_dist))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    fsim_Y = ((2 * mu_Y_ref * mu_Y_dist + c1) * (2 * sigma_Y_ref_dist + c2)) / (
                (mu_Y_ref ** 2 + mu_Y_dist ** 2 + c1) * (sigma_Y_ref ** 2 + sigma_Y_dist ** 2 + c2))
    fsim_Cb = ((2 * mu_Cb_ref * mu_Cb_dist + c1) * (2 * sigma_Cb_ref_dist + c2)) / (
                (mu_Cb_ref ** 2 + mu_Cb_dist ** 2 + c1) * (sigma_Cb_ref ** 2 + sigma_Cb_dist ** 2 + c2))
    fsim_Cr = ((2 * mu_Cr_ref * mu_Cr_dist + c1) * (2 * sigma_Cr_ref_dist + c2)) / (
                (mu_Cr_ref ** 2 + mu_Cr_dist ** 2 + c1) * (sigma_Cr_ref ** 2 + sigma_Cr_dist ** 2 + c2))

    fsim_score = (fsim_Y + fsim_Cb + fsim_Cr) / 3.0

    return fsim_score