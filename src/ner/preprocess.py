from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from spellchecker import SpellChecker
import re


def text_processing(raw_data):
    """Process raw ocr output to
    make suitable for ner model"""
    regex = "[^a-zA-Z0-9.,]+"
    filtered_text = re.sub(regex, " ", str(raw_data))
    spell = SpellChecker()
    processed_text = []
    for sent in sent_tokenize(filtered_text):
        processed = []
        for word in word_tokenize(sent):
            processed.append(spell.correction(word))
        for _, pos in pos_tag(processed):
            if pos == "NNP":
                processed_text.append(" ".join(processed))
                break

        # if any([True for _, pos in pos_tag(processed) if pos[:3] == "NNP"]):
        #     processed_text.append(" ".join(processed))

    return "".join(processed_text)


if __name__ == "__main__":
    with open(
        "C:\\Users\\Divakar.Sharma\\Desktop\\Travelers\\AI-Underwriting\\src\\ner"
        "\\data\\ocr_out_text_sample_2.txt"
    ) as data:
        raw_data = data.read()
        # print("raw_dataset", raw_data)
        processed_data = text_processing(raw_data)
        print("processed_dataset", processed_data)

    with open(
        "C:\\Users\\Divakar.Sharma\\Desktop\\Travelers\\AI-Underwriting\\src\\ner\\data\\new.csv",
        "w",
    ) as new:
        new.write(processed_data)
