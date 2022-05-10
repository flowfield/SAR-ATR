import cv2
import numpy as np
import torch
import os

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from subprocess import Popen


def enhance_constrast(img):
    clahe_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(11, 11))
    img_out = clahe_img.apply(img)

    return img_out


def mstar2tif(img_path):
    a_path = []

    folderpaths_a = [os.path.join(img_path, file_name) for file_name in os.listdir(img_path)]
    a_path.extend(folderpaths_a)

    for i in range(len(a_path)):
        org_path = a_path[i]
        save_path = org_path[-7:]

        Popen(r'"./datasets/MSTAR/mstar2tiff.exe" -i ' + org_path + ' -o ' + save_path + '.tif', shell=True)


def mstar2jpg(img_path):
    a_path = []

    folderpaths_a = [os.path.join(img_path, file_name) for file_name in os.listdir(img_path)]
    a_path.extend(folderpaths_a)

    for i in range(len(a_path)):
        org_path = a_path[i]
        save_path = org_path[-7:]

        Popen(r'"./datasets/MSTAR/mstar2jpg.exe" -i ' + org_path + ' -o ' + save_path + '.jpg', shell=True)


def enl(img):
    img = torch.tensor(img, dtype=torch.float)
    mu = torch.mean(img)
    var = torch.var(img)

    ENL = (mu ** 2) / var
    return ENL


def psnr(img1, img2):
    maxI = torch.tensor(255.0, dtype=torch.float)

    img1 = torch.tensor(img1, dtype=torch.float) * maxI
    img2 = torch.tensor(img2, dtype=torch.float)

    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(maxI) - 10 * torch.log10(mse)


def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def cfar2d(inputImg, output_path, GUARD_CELLS, BG_CELLS, ALPHA):
    CFAR_UNITS = 1 + (GUARD_CELLS * 2) + (BG_CELLS * 2)

    estimateImg = np.zeros((inputImg.shape[0], inputImg.shape[1], 1), np.uint8)

    for i in range(inputImg.shape[0] - CFAR_UNITS):
        center_cell_x = i + BG_CELLS + GUARD_CELLS
        for j in range(inputImg.shape[1] - CFAR_UNITS):
            center_cell_y = j + BG_CELLS + GUARD_CELLS
            average = 0
            for k in range(CFAR_UNITS):
                for l in range(CFAR_UNITS):
                    if (k >= BG_CELLS) and (k < (CFAR_UNITS - BG_CELLS)) and (l >= BG_CELLS) and (
                            l < (CFAR_UNITS - BG_CELLS)):
                        continue
                    average += inputImg[i + k, j + l]
            average /= (CFAR_UNITS * CFAR_UNITS) - (((GUARD_CELLS * 2) + 1) * ((GUARD_CELLS * 2) + 1))

            if inputImg[center_cell_x, center_cell_y] > (average * ALPHA):
                estimateImg[center_cell_x, center_cell_y] = 255

    # output
    cv2.imwrite(output_path, estimateImg)

    return estimateImg
