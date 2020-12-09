"""
@Author:TeJas.Lotankar

Description
-----------
	Main function for passing image from ocr engine.
	Here you can call pre-processing function as per reuirememnt and pass into
	OCR engine.
"""

# imports
import numpy as np
import cv2
from PIL import Image
import os

# tesseract
import pytesseract

# os.environ["TESSDATA_PREFIX"] = r"C:\Users\Divakar.Sharma\AppData\Local\Tesseract-OCR"
# pytesseract.pytesseract.tesseract_cmd = 'C:/Users/Divakar.Sharma/AppData/Local/Tesseract-OCR/tesseract.exe'

# Kraken
from kraken.lib.models import load_any
from kraken import rpred, binarization
from kraken.pageseg import segment

# EasyOCR
import easyocr

from .preprocessing_utils import (
    get_grayscale_img,
    get_thresh_adaptive,
    get_fastnNlMeanDenoising,
)

# ----------------------- Other_Utils -----------------------------------


def save_as_file(input_text, path_to_save, name_of_file):
    """
    Description
    -----------
            Helper function for savinf output into file.

    Params
    ------
            input_text : (str)
                    Input text to be saved in the file.
            path_to_save : (str)
                    path where file to be saved.
            name_of_file : (str)
                    name of file with extension.

    Returns
    -------
            output_dir : (str)
                    Path to the saved file.
    """
    output_dir = os.path.join(path_to_save, name_of_file)

    if not os.path.isdir(path_to_save):
        os.makedirs(path_to_save)
    with open(output_dir, "w") as fp:
        fp.write(input_text)

    return output_dir


def prepare_image_for_tesseract(img_proc):
    """
    Pre-processing functions are called here and stacked.
    Image will go through here once before OCR.
    Create pre-processing pipeline here
    """
    img_proc = get_grayscale_img(img_proc)
    img_proc = get_fastnNlMeanDenoising(img_proc)
    img_proc = get_thresh_adaptive(img_proc)

    return img_proc


# ----------------------- pyTesseract_OCR_Code -----------------------------


def get_ocr_tesseract(img_path):
    """
    Description
    -----------
    Gets image path and pre-process, then passes to Tesseract OCR engine.

    Params
    ------
    img_path : (str)
            Path to the image to be processed.
    output_as_file : (boolean) False
            If set True, function will output path to the file contening ocr text.

    Returns
    -------
    ocr_text : str
            OCR results.
    """
    # print("Tesseract Running...")
    img_cv2_obj = cv2.imread(img_path)

    # img_proc = prepare_image_for_tesseract(img_cv2_obj)

    # converting into PIL image format
    img_pil_obj = Image.fromarray(img_cv2_obj)

    # passing to tesseract
    ocr_text = pytesseract.image_to_string(img_pil_obj)
    print("ocr_text is being generated", ocr_text)

    return ocr_text


# ---------------------- Kraken_OCR ---------------------------


def get_ocr_kraken(img_path):
    """
    Description
    -----------
    Gets image path and pre-process, then passes to Kraken OCR engine.

    Params
    ------
    img_path : str
            Path to the image to be processed.

    Returns
    -------
    ocr_text : str
            OCR results.
    """
    print("Kraken Running...")
    img_pil_obj = Image.open(img_path)

    # Binarization
    gen_image = binarization.nlbin(img_pil_obj)

    # Loading Model
    kraken_model = load_any("src/ocr/models/en_best.mlmodel")

    # page segmentation
    bound = segment(gen_image)

    # Calling api for prediction
    genrator = rpred.rpred(network=kraken_model, im=gen_image, bounds=bound)

    ocr_text = "\n".join(pr.prediction for pr in genrator)

    return ocr_text


# ----------------------- EasyOCR -------------------------------


def get_easyocr(img_path):
    """
    Description
    -----------
    Gets image path and pre-process, then passes to EasyOCR engine.

    Params
    ------
    img_path : str
            Path to the image to be processed.

    Returns
    -------
    output_dir : str
            Path to the text file contening OCR results.
    """
    print("EasyOCR Running...")
    img_cv2_obj = cv2.imread(img_path)

    reader = easyocr.Reader(["en"], gpu=False)
    ocr_text = reader.readtext(img_cv2_obj, detail=0, paragraph=True)

    return ocr_text
