"""
@Author:TeJas.Lotankar

Description
-----------
	Utils required for pre-processing images before passing into OCR engine are written here.
"""

# imports
import cv2
import numpy as np
from pdf2image import convert_from_path
import os

# get grayscale image
def get_grayscale_img(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# shadow Removing
def shadow_removal(image):
    # img = cv2.imread(path,-1)
    rgb_planes = cv2.split(image)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)  # Removes noise from the image
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(
            diff_img,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8UC1,
            dst=None,
        )
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes)
    return result_norm


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 3)


# Histogram equalization
def equalize_hist(img_path, clip_limit=2.0):
    img = img_path
    # Creating object for CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    # Applying equalization
    outImg = clahe.apply(img)

    return outImg


# thresholding
def get_thresh_adaptive(img, thresh=255):
    ada_th = cv2.adaptiveThreshold(
        img, thresh, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    return ada_th


# dilation
def dilate(image):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def get_fastnNlMeanDenoising(img, templateWindowSize=7, searchWindowSize=21, h=4):
    outImg_cv2 = cv2.fastNlMeansDenoising(
        img, None, templateWindowSize, searchWindowSize, h
    )
    return outImg_cv2


# PDF to Image converter
# def pdf_to_img(pdf_path, to_grayscale=False, output_folder=None):

#     page_count = 0

#     if output_folder == None:
#         output_folder = os.path.join(
#             *pdf_path.split("/")[:-1], pdf_path.split("/")[-1].split(".")[-2]
#         )

#     os.makedirs(output_folder)

#     images = convert_from_path(
#         pdf_path,
#         grayscale=to_grayscale,
#         output_folder=output_folder,
#         fmt="jpeg",
#         dpi=300,
#     )

# for img in images:
# 	img.save(os.path.join(output_folder, "page_{}.jpg".format(page_count)), 'JPEG')
# 	page_count+=1

# print("PDF to images task completed..")
