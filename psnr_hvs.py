import cv2
import numpy as np


def psnr_hvs(img1, img2):
    # Convert images to float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Compute the squared error
    mse = np.mean((img1 - img2) ** 2)

    # Compute the maximum pixel value
    max_pixel = 255.0

    # Compute PSNR-HVS
    psnr_hvs = 10 * np.log10(max_pixel ** 2 / mse)

    return psnr_hvs

