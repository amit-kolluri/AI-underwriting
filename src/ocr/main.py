'''
@Auther:TeJas.Lotankar

Description
-----------
	Main OCR function.
'''



import cv2
import os
import json

from .ocr_utils import get_ocr_tesseract, get_ocr_kraken, get_easyocr
from .ocr_utils import save_as_file



from .preprocessing_utils import (
    get_grayscale_img,
    get_thresh_adaptive,
    get_fastnNlMeanDenoising,
    remove_noise,
    equalize_hist,
)

def ocr_main(file_path, ocr_engine="tesseract", output_as_file=False):
	'''
	Description
	-----------
		Accepts single or multiple images and returns ocr text.

	Params
	-----
		file_path: (str)
			Path to single image file or folder containing multiple images.
		ocr_engine : (str) "tesseract"
			Select OCR engine for processing. Default to pyTeseract.
			- "tesseract" : pyTesseract
			- "kraken" : Kraken OCR
			- "easyocr" : EasyOCR
		output_as_file : (boolean) False 
			If set True, function will output path to the file contening ocr text.

	Returns
	-------
		output_dir : str or list of str
		Path or list of paths for output text files.
	'''
	ocr_engine_map = {
		'tesseract': get_ocr_tesseract,
		'kraken': get_ocr_kraken,
		'easyocr': get_easyocr
	}

	ocred_text = ocr_engine_map[ocr_engine](file_path)

	if output_as_file:
		path_to_save = os.path.join(*[pth for pth in file_path.split("/")[:-1]])
		file_proc_name = file_path.split("/")[-1]
		out_dir = save_as_file(ocred_text, path_to_save, file_proc_name+"_OCR_text.txt")
		return out_dir

	return ocred_text
	
def prep_image(
    proc_image,
    steps):
    """
    Description
    -----------
            Pipeline for processing image.
    """
    # Setting up directory
    WORKING_DIR = "./prep_img-files/"
    if not os.path.isdir(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    return_list = list()
    for step in steps:
        if step == "gray_scale":
            step_dict = {}
            step_dict["step"] = "gray_scale"
            proc_image = get_grayscale_img(proc_image)
            step_dict["image"] = proc_image
            return_list.append(step_dict)

        if step == "histogram_equalization":
            step_dict = {}
            step_dict["step"] = "histogram_equalization"
            proc_image = equalize_hist(proc_image)
            step_dict["image"] = proc_image
            return_list.append(step_dict)

        if step == "binarization":
            step_dict = {}
            step_dict["step"] = "binarization"
            proc_image = get_thresh_adaptive(proc_image)
            step_dict["image"] = proc_image
            return_list.append(step_dict)

        if step == "fast_and_mean_desnoise":
            step_dict = {}
            step_dict["step"] = "fast_and_mean_desnoise"
            proc_image = get_fastnNlMeanDenoising(proc_image)
            step_dict["image"] = proc_image
            return_list.append(step_dict)

        if step == "denoising":
            step_dict = {}
            step_dict["step"] = "denoising"
            proc_image = remove_noise(proc_image)
            step_dict["image"] = proc_image
            return_list.append(step_dict)
    return return_list


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
