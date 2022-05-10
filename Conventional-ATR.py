import os

import cv2
from skimage import restoration
from matplotlib import pyplot as plt

from ConventionalATRFunctions import *


## image open
img_path = './Santos_SAR_HV.tif'
img_path = ''
org_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
gray_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)

# cfar_image = cfar2d(img_path)
# noise_img = random_noise(gray_img, mode='speckle', var=0.05)

## Reconstruction or Filtering
median_img = cv2.medianBlur(gray_img, 3)
bilateral_img = cv2.bilateralFilter(gray_img, -1, sigmaColor=50, sigmaSpace=50)
lee_img = lee_filter(gray_img, 3)
wavelet_img = restoration.denoise_wavelet(gray_img, convert2ycbcr=False, method='BayesShrink', mode='soft',
                                          rescale_sigma=True)
guided_img = cv2.ximgproc.guidedFilter(median_img, gray_img, 33, 2, -1)

fig = plt.figure()
rows = 2
cols = 2

ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(gray_img, cmap='gray')
ax1.axis("off")

ax2 = fig.add_subplot(rows, cols, 2)
ax2.imshow(median_img, cmap='gray')
ax2.axis("off")

ax3 = fig.add_subplot(rows, cols, 3)
ax3.imshow(bilateral_img, cmap='gray')
ax3.axis("off")

ax4 = fig.add_subplot(rows, cols, 4)
ax4.imshow(guided_img, cmap='gray')
ax4.axis("off")

plt.tight_layout()
plt.show()

org_enl = enl(gray_img)
med_enl = enl(median_img)
bilat_enl = enl(bilateral_img)
wave_enl = enl(wavelet_img)

med_psnr = psnr(median_img, gray_img)
bilat_psnr = psnr(bilateral_img, gray_img)
wave_psnr = psnr(wavelet_img, gray_img)

print('ENL')
print(f'org:{org_enl:9.6f}')
print(f'median:{med_enl:9.6f}')
print(f'bilateral:{bilat_enl:9.6f}')
print(f'wavelet:{wave_enl:9.6f}')

print('\nPSNR')
print(f'median:{med_psnr:9.6f}')
print(f'bilateral:{bilat_psnr:9.6f}')
print(f'wavelet:{wave_psnr:9.6f}')

# ## Thresholding
# threshold_mean = cv2.adaptiveThreshold(median_img, 255,
#                                        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
# threshold_gaussian = cv2.adaptiveThreshold(median_img, 255,
#                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
#
# fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
#
# axes[0].imshow(threshold_mean, cmap=plt.cm.gray)
# axes[0].set_title('Mean Threshold')
#
# axes[1].imshow(threshold_gaussian, cmap=plt.cm.gray)
# axes[1].set_title('Gaussian Threshold')
#
# for ax in axes:
#     ax.axis('off')
#
# plt.tight_layout()
# plt.show()

## CFAR
output_path = './HV_CFAR.tif'
GUARD_CELLS = 5
BG_CELLS = 10
ALPHA = 2
cfar_img = cfar2d(median_img, output_path, GUARD_CELLS, BG_CELLS, ALPHA)

## Edge Detect
# img_mean, img_dev = cv2.meanStdDev(bilateral_img)
# img_mean2, img_dev2 = cv2.meanStdDev(median_img)
# tau = 2
#
# img_blackwhite = bilateral_img
# img_blackwhite2 = median_img
# w, h = img_blackwhite.shape
# w2, h2 = img_blackwhite2.shape
#
# for i in range(w):
#     for j in range(h):
#         if img_blackwhite[i][j] <= (img_mean + tau*img_dev):
#             img_blackwhite[i][j] = 0
#         else:
#             img_blackwhite[i][j] = 255
#
# for i in range(w2):
#     for j in range(h2):
#         if img_blackwhite2[i][j] <= (img_mean2 + tau*img_dev2):
#             img_blackwhite2[i][j] = 0
#         else:
#             img_blackwhite2[i][j] = 255
#
# img_laplacian = cv2.Laplacian(img_blackwhite, cv2.CV_8U, ksize=3)
# img_sobel = cv2.Sobel(img_blackwhite, cv2.CV_8U, 1, 1, ksize=3)
# img_canny = cv2.Canny(img_blackwhite, threshold1=0.9, threshold2=1, apertureSize=3)
#
# img_laplacian2 = cv2.Laplacian(img_blackwhite2, cv2.CV_8U, ksize=3)
# img_sobel2 = cv2.Sobel(img_blackwhite2, cv2.CV_8U, 1, 1, ksize=3)
# img_canny2 = cv2.Canny(img_blackwhite2, threshold1=0.9, threshold2=1, apertureSize=3)
#
# fig = plt.figure()
# rows = 2
# cols = 3
#
# ax1 = fig.add_subplot(rows, cols, 1)
# ax1.imshow(img_blackwhite, cmap='gray')
# ax1.axis("off")
#
# ax2 = fig.add_subplot(rows, cols, 2)
# ax2.imshow(img_laplacian, cmap='gray')
# ax2.axis("off")
#
# ax3 = fig.add_subplot(rows, cols, 3)
# ax3.imshow(img_canny, cmap='gray')
# ax3.axis("off")
#
# ax4 = fig.add_subplot(rows, cols, 4)
# ax4.imshow(img_blackwhite2, cmap='gray')
# ax4.axis("off")
#
# ax5 = fig.add_subplot(rows, cols, 5)
# ax5.imshow(img_laplacian2, cmap='gray')
# ax5.axis("off")
#
# ax6 = fig.add_subplot(rows, cols, 6)
# ax6.imshow(img_canny2, cmap='gray')
# ax6.axis("off")
#
# plt.tight_layout()
# plt.show()

## Morphology
# 구조화 요소 커널, 사각형 생성
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

image_morphology = cv2.morphologyEx(cfar_img, cv2.MORPH_CLOSE, k)
image_morphology = cv2.morphologyEx(image_morphology, cv2.MORPH_OPEN, k)

plt.figure()
plt.imshow(image_morphology, cmap='gray')

plt.tight_layout()
plt.show()

## Bounding Box
if np.max(cfar_img) < 2:
    image_bbox = cfar_img * 255
else:
    image_bbox = cfar_img
image_bbox_cpy = image_bbox.copy()

cnts = cv2.findContours(image_bbox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(cnts) == 2:
    cnts = cnts[0]
else:
    cnts = cnts[1]

image_bbox_cpy = cv2.cvtColor(image_bbox_cpy, cv2.COLOR_GRAY2BGR)
bbox_margin = 10
for contour in cnts:
    area = cv2.contourArea(contour)
    if area > 16:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_bbox_cpy, (x - bbox_margin, y - bbox_margin),
                      (x + w + bbox_margin, y + h + bbox_margin), (0, 0, 255), 1)

plt.figure()
plt.imshow(image_bbox_cpy)

plt.tight_layout()
plt.show()

## N MSTAR into clutter
clutter_path = []
path_clutter = './datasets/MSTAR/CLUTTER/15_DEG/JPGs/'
clutter_folderpath = [os.path.join(path_clutter, filename) for filename in os.listdir(path_clutter)]
clutter_path.extend(clutter_folderpath)

random_clutter = np.random.randint(0, len(clutter_path), 1)

org_clutter = cv2.imread(clutter_path[random_clutter[0]], cv2.IMREAD_GRAYSCALE)
org_clutter = enhance_constrast(org_clutter)
cpy_clutter = org_clutter.copy()
clutter_w, clutter_h = org_clutter.shape

mstar_path = []
base_path = './datasets/flowers/flower_photos/1_ 2S1 [158 x 158]/'
mstar_folderpath = [os.path.join(base_path, filename) for filename in os.listdir(base_path)]
mstar_path.extend(mstar_folderpath)

num_sar_img = 5
random_x = np.random.randint(0, clutter_w - 150, num_sar_img)
random_y = np.random.randint(0, clutter_h - 150, num_sar_img)
random_img = np.random.randint(0, len(mstar_path), num_sar_img)

for i in range(num_sar_img):
    mstar_img = cv2.imread(mstar_path[i], cv2.IMREAD_GRAYSCALE)
    ret, mask = cv2.threshold(mstar_img, 255*0.7, 255, cv2.THRESH_BINARY_INV)

    mask_inv = cv2.bitwise_not(mask)
    mask_w, mask_h = mask.shape

    x = random_x[i]
    y = random_y[i]

    clutter_cut = cpy_clutter[x:x+mask_w, y:y+mask_h]

    img1 = cv2.bitwise_and(mstar_img, mstar_img, mask=mask_inv)
    img2 = cv2.bitwise_and(clutter_cut, clutter_cut, mask=mask)

    temp = cv2.add(img1, img2)
    cpy_clutter[x:x + mask_w, y:y + mask_h] = temp

plt.figure()
plt.imshow(cpy_clutter, cmap='gray')

plt.tight_layout()
plt.show()

## SIFT, ORB
orb = cv2.ORB_create(nfeatures=30)

keypoints, descriptors = orb.detectAndCompute(bilateral_img, None)

print('Descriptor.shape: ', descriptors.shape)

draw = cv2.drawKeypoints(bilateral_img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure()
plt.imshow(draw)

plt.tight_layout()
plt.show()