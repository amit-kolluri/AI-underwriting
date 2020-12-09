import json
import streamlit as st
from streamlit import caching
from PIL import Image
from src.utils import _pdf2images, _save_pdf, _save_ocr_text, _save_results
from src.ner import main as ner_main
# from src.ocr import ocr_main
from src.ocr.ocr_streamlit_main import main_app

image = Image.open("images/logo/yash_logo.png")
rgb_im = image.convert("RGB")

st.sidebar.image(rgb_im, use_column_width=False)

st.markdown(
    "<h2 style='text-align: center; color: black;'>Traveler BSI</h2>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Choose a PDF...", type="pdf")
print(uploaded_file)
# caching.clear_cache()
if uploaded_file is not None:
    # caching.clear_cache()
    attachment_path, file_name = _save_pdf(uploaded_file)
    image_list = _pdf2images(attachment_path, file_name)
    attachment_dict = {
        "file_name_original": "file_name_original",
        "file_name": file_name,
        "file_path": attachment_path,
        "images": [],
    }
    for number, image in enumerate(image_list):
        image_dict = {
            "image_name": image["image_name"],
            "image_path": image["image_path"],
            "image_number": image["image_number"],
        }
        # raw_text = ocr_main.ocr_main(image["image_path"])
        print("image path:", image["image_path"])
        main_app(image["image_path"])
        with open("src/ocr/ocr_results_temp/ocr_data.json", "r") as ocr_json:
            ocr_data = json.load(ocr_json)
        print("json data:", ocr_data)
        # raw_text = st.selectbox("Select OCR output for NER:", ["tesseract", "kraken", "easyocr"],
        #                    key=number)

        for text in ocr_data["ocr_output"]:
            if text["ocr_engine"]:
                raw_text = text["raw_text"]
                break

        image_dict["image_text_raw"] = raw_text
        st.markdown("---")
        steps = st.multiselect(
            "Select preprocessing steps:",
            ["regex", "spell correction", "important sentences"],
            key=number,
        )
        print("steps for preprocessing:", steps)
        if st.button("Data Prep"):
            image_dict["image_text_process"] = ner_main.demo_preprocessing(
                raw_text, steps
            )
            print("output:", image_dict["image_text_process"])
            st.write(image_dict["image_text_process"], key=number)

        _save_ocr_text(image["image_name"], raw_text)
        ner_model_path = "src/ner/models/"
        model = st.selectbox(
            "Select NER model for prediction:",
            ["spacy_small", "spacy_med", "bert", "crf", "crf_lstm", "diet"],
            key=number,
        )
        if st.button("Run Model"):
            results = ner_main.inference(raw_text, ner_model_path, model)
            st.write(results, key=number)
            image_dict["ner"] = results
            attachment_dict["images"].append(image_dict)

        break
        # if st.button("Next page"):
        #     continue
        # else:
        #     break
    result = _save_results(file_name, attachment_dict)
