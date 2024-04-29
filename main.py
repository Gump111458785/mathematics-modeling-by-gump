import cv2

import fsim
import psnr
import sharpness
import y_psnr
import ssim
import psnr_hvs
import psnr_hvsm
import psnr_ha
import psnr_hma
import vif
import nqm
import wsnr
import vsnr
import fsim
import fsimc
import numpy as np
img1 = cv2.imread('reference_images/I01.BMP')
img2 = cv2.imread('distorted_images/I01_01_1.bmp')
cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.waitKey(0)
#峰值信噪比
print(psnr.psnr(img1,img2))
#为亮度分量计算的峰值信噪比
print(y_psnr.psnr_luminance(img1,img2,5))#scale为边界裁剪值，避免边界误差，此处默认5像素
#ssim指标
print(ssim.calculate_ssim(img1,img2))
#mssim指标
print(ssim.mssim(img1,img2))
#psnr_hvs指标
print(psnr_hvs.psnr_hvs(img1,img2))
#psnr_hvs_m指标
print(psnr_hvsm.psnr_hvs_m(img1,img2))
#psnr_ha指标
print(psnr_ha.psnr_ha(img1,img2))
#psnr_hma指标
print(psnr_hma.psnr_hma(img1,img2))
#vif指标
print(vif.vif(img1,img2))
#nqm指标
print(nqm.nqm(img1,img2))
#wsnr指标
print(wsnr.compute_wsnr(img1,img2))
#vsnr指标
print(vsnr.vsnr(img1,img2))
#fsim指标
print(fsim.fsim(img1,img2))
#fsimc指标
print(fsimc.fsim_color(img1,img2))
#锐度检测
print(sharpness.sharpness(img1,img2))

cv2.destroyAllWindows()