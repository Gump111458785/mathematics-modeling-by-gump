import cv2
import numpy as np

def psnr_luminance(target, ref, scale):
    # 转换为YCbCr颜色空间
    target_y = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    ref_y = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    # 根据指定的尺寸大小裁剪图像
    target_y = target_y[scale:-scale, scale:-scale]
    ref_y = ref_y[scale:-scale, scale:-scale]

    # 计算差异
    diff = ref_y - target_y

    # 计算均方误差（MSE）
    mse = np.mean(diff ** 2)

    # 计算PSNR
    if mse == 0:
        return float('inf')
    else:
        return 20 * np.log10(255.0 / np.sqrt(mse))


