import cv2
import numpy as np

def nqm(image_ref, image_dist):
    # 将图像转换为灰度图像
    image_ref_gray = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
    image_dist_gray = cv2.cvtColor(image_dist, cv2.COLOR_BGR2GRAY)

    # 将图像划分为大小相等的块
    block_size = 8
    h, w = image_ref_gray.shape
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size

    # 计算每个块的块内平均值
    ref_block_means = []
    dist_block_means = []
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            ref_block = image_ref_gray[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            dist_block = image_dist_gray[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            ref_block_mean = np.mean(ref_block)
            dist_block_mean = np.mean(dist_block)
            ref_block_means.append(ref_block_mean)
            dist_block_means.append(dist_block_mean)

    # 计算块间差异
    block_diffs = np.abs(np.array(ref_block_means) - np.array(dist_block_means))

    # 归一化处理得到最终的 NQM 分数
    nqm_score = np.mean(block_diffs) / 255.0

    return nqm_score


