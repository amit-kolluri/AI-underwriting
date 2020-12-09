"""
@Author:TeJas.Lotankar

Description
-----------
	Streamlit Utils.
"""

import cv2
import os
import json
import streamlit as st

from .preprocessing_utils import (
    get_grayscale_img,
    get_thresh_adaptive,
    get_fastnNlMeanDenoising,
    remove_noise,
    equalize_hist,
    opening,
    dilate,
    shadow_removal,
)

from .ocr_main import ocr_main


def prep_image(
    img_obj,
    to_remove_shadow=False,
    to_grayscale=False,
    suggested_prep=False,
    to_binarize=False,
    to_fast_n_mean_denoise=False,
    to_denoise=False,
    to_opening=False,
    to_dilate=False,
):
    """
    Description
    -----------
            Pipeline for processing image.
    """
    # Setting up directory
    WORKING_DIR = "src/ocr/prep_img-files/"
    if not os.path.isdir(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    # steps_done = list()
    proc_image = img_obj

    if suggested_prep:
        # proc_image = shadow_removal(proc_image)
        proc_image = get_grayscale_img(proc_image)
        proc_image = get_thresh_adaptive(proc_image)
        proc_image = get_fastnNlMeanDenoising(proc_image)

        out_path = os.path.join(WORKING_DIR, "prep_img.jpg")
        cv2.imwrite(out_path, proc_image)

        return proc_image, out_path

    if to_grayscale:
        proc_image = get_grayscale_img(proc_image)

    if to_denoise:
        proc_image = remove_noise(proc_image)

    if to_dilate:
        proc_image = dilate(proc_image)

    if to_opening:
        proc_image = opening(proc_image)

    if to_binarize:
        proc_image = get_thresh_adaptive(proc_image)

    if to_remove_shadow:
        proc_image = shadow_removal(proc_image)

    if to_fast_n_mean_denoise:
        proc_image = get_fastnNlMeanDenoising(proc_image)

    if to_denoise:
        proc_image = remove_noise(proc_image)

    out_path = os.path.join(WORKING_DIR, "prep_img.jpg")
    cv2.imwrite(out_path, proc_image)

    # return json.dumps({"image_path": out_path, "steps": steps_done})
    return proc_image, out_path


def get_ocr(img_path, ocrs=["tesseract"]):
    """
    Description
    -----------
            Pipeline for OCR.
    """
    ocr_out = list()
    for ocr in ocrs:
        ocr_text = ocr_main(img_path, ocr)
        ocr_out.append({"ocr_engine": ocr, "raw_text": ocr_text})

    return ocr_out


# im = cv2.imread("Battery_Stilwell_Agreement/test_img_01_cropped.jpg")
# a = prep_image(im, True, False, True, True)
# # b = get_ocr("Battery_Stilwell_Agreement/test_img_01_cropped.jpg", ["tesseract", "kraken", "easyocr"])
# print(a)
