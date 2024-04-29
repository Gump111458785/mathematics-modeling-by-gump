import cv2
import numpy as np
import pywt


def psnr_ha(img1, img2):
    # Convert images to float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Compute the squared error
    mse = np.mean((img1 - img2) ** 2)

    # Compute the maximum pixel value
    max_pixel = 255.0

    # Compute PSNR-HA using Haar transform
    coeffs1 = pywt.dwt2(img1, 'haar')
    coeffs2 = pywt.dwt2(img2, 'haar')
    cA1, (cH1, cV1, cD1) = coeffs1
    cA2, (cH2, cV2, cD2) = coeffs2

    mse_ha = (mse + np.mean((cH1 - cH2) ** 2) + np.mean((cV1 - cV2) ** 2) + np.mean((cD1 - cD2) ** 2)) / 4

    # Compute PSNR-HA
    psnr_ha = 10 * np.log10(max_pixel ** 2 / mse_ha)

    return psnr_ha
