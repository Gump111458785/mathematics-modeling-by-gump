import numpy as np
import cv2
def psnr_hvs_m(img1, img2):
    # Compute the squared error
    mse = np.mean((img1 - img2) ** 2)

    # Compute the maximum pixel value
    max_pixel = 255.0

    # Compute PSNR-HVS-M
    psnr_hvs_m = 10 * np.log10(max_pixel ** 2 / mse)

    return psnr_hvs_m


