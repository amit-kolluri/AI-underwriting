"""
@Auther:TeJas.Lotankar

Description
-----------
	Main OCR function.
"""

import os

from .ocr_utils import get_ocr_tesseract, get_ocr_kraken, get_easyocr
from .ocr_utils import save_as_file
from tqdm import tqdm


def ocr_main(file_path, ocr_engine="tesseract", output_as_file=False):
    """
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
    """
    ocr_engine_map = {
        "tesseract": get_ocr_tesseract,
        "kraken": get_ocr_kraken,
        "easyocr": get_easyocr,
    }

    ocred_text = ocr_engine_map[ocr_engine](file_path)

    if output_as_file:
        path_to_save = os.path.join(*[pth for pth in file_path.split("/")[:-1]])
        file_proc_name = file_path.split("/")[-1]
        out_dir = save_as_file(
            ocred_text, path_to_save, file_proc_name + "_OCR_text.txt"
        )
        return out_dir

    return ocred_text

    # ===== for multiple files in folder ======
    # output_dir = list()

    # if os.path.isdir(file_path):
    # 	for img_file in tqdm(os.listdir(file_path)):
    # 		out_file_path = get_ocr_tesseract(os.path.join(file_path,img_file))
    # 		output_dir.append(out_file_path)
    # else:
    # 	ocred_text = get_ocr_tesseract(file_path)

    # return output_dir


# outs = ocr_main("Battery_Stilwell_Agreement/Testing01/c604478c-ab24-49b9-a45d-2f71ae644098-05.jpg", "easyocr")
# print(outs)

# out = get_ocr_kraken("Battery_Stilwell_Agreement/Testing01/c604478c-ab24-49b9-a45d-2f71ae644098-05.jpg")
# print(out)
