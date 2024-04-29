import math
import numpy as np


def psnr(img1, img2):
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr2 = 20 * math.log10(1 / math.sqrt(mse))
    return psnr2


