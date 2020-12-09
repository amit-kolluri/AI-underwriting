"""
@Author:TeJas.Lotankar

Description
-----------
	Streamlit main.
"""

# imports
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os

from .ocr_streamlit_utils import prep_image, get_ocr


def main_app(inp_img):
    # setting up directory
    WORKING_DIR = "src/ocr/ocr_results_temp/"
    if not os.path.isdir(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    #
    # inp_img = st.file_uploader(
    #     "Upload your image, (jpg, png)", type=["jpg", "png", "jpeg"]
    # )

    if inp_img:
        img_pil = Image.open(inp_img)
        img_cv2 = np.array(img_pil)

        st.image(img_pil, caption="Original Image", use_column_width=True)
        # st.markdown("---")

        suggested_prep = st.checkbox("Recomended")

        col_0 = st.beta_columns(3)
        to_remove_shadow = col_0[0].checkbox("Remove_Shadow")
        to_grayscale = col_0[1].checkbox("Grayscale")
        to_binarize = col_0[2].checkbox("Binarize")

        col_1 = st.beta_columns(3)
        to_fast_n_mean_denoise = col_1[0].checkbox("Fast_n_Mean_Denoise")
        to_denoise = col_1[1].checkbox("De-Noising")
        to_opening = col_1[2].checkbox("Morphological_Opening")

        col_2 = st.beta_columns(3)
        to_dilate = col_2[0].checkbox("Dilation")

        if st.button("Process_Image"):
            out_img, out_img_path = prep_image(
                img_cv2,
                to_remove_shadow,
                to_grayscale,
                suggested_prep,
                to_binarize,
                to_fast_n_mean_denoise,
                to_denoise,
                to_opening,
                to_dilate,
            )
            # st.image(out_img, caption="Processed image", use_column_width=True)
            # Placing images
            im1, im2 = st.beta_columns(2)
            im1.header("Original")
            im1.image(img_pil, use_column_width=True)
            im2.header("Processed")
            im2.image(out_img, use_column_width=True)
            OCR_MARKER = True

    # OCR handing
    st.markdown("---")
    ocr_en = st.multiselect("Select OCR Engine..", ["tesseract", "kraken", "easyocr"])
    if st.button("Process OCR"):
        ocr_out = get_ocr("src/ocr/prep_img-files/prep_img.jpg", ocr_en)

        # Dumping OCR to json
        ocr_json_object = json.dumps({"ocr_output": ocr_out})
        ocr_out_path = os.path.join(WORKING_DIR, "ocr_data.json")
        with open(ocr_out_path, "w") as fp:
            fp.write(ocr_json_object)

        txt_col = st.beta_columns(int(len(ocr_en)))
        for i in range(0, int(len(ocr_en))):
            ocr_txt = ocr_out[i].get("raw_text")
            # print(type(ocr_txt))
            if isinstance(ocr_txt, list):
                # print("Came Here........")
                ocr_txt_out = "\n".join(i for i in ocr_txt)
                ocr_txt = ocr_txt_out
            txt_col[i].header(ocr_out[i].get("ocr_engine"))
            txt_col[i].text(ocr_txt)
            # print(ocr_out[i].get("raw_text"))
        # st.text(ocr_out[0].get("raw_text"))

    # return ocr_out_path

#
# main_app()
