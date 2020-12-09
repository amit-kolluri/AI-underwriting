# -*- coding: utf-8 -*-
from .preprocess import text_processing
from .bert.predict import bert_prediction
from .spacy.predict import spacy_prediction
from .crf_lstm.predict import crf_lstm_prediction
from .crf.predict import crf_prediction
from .diet.predict import diet_prediction
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from spellchecker import SpellChecker
import re


def demo_preprocessing(raw_text, steps):
    """
    Parameters
    ----------
    raw_text
    steps

    Returns
    -------
    processed text

    """
    regex = "[^a-zA-Z0-9 .]+"
    # preprocessed = []
    for step in steps:
        # preprocess = {}
        print("print step:", step)
        if step == "regex":
            # preprocess["step"] = "regex"
            filtered_text = re.sub(regex, " ", raw_text)
            raw_text = filtered_text
            # preprocess["processed_text"] = filtered_text
            # preprocessed.append(preprocess)
        if step == "spell correction":
            sentence = []
            # preprocess["step"] = "spell correction"
            tokenized_text = word_tokenize(raw_text)
            for word in tokenized_text:
                spell = SpellChecker()
                sentence.append(spell.correction(word))
                raw_text = " ".join(sentence)
        if step == "Important Sentences":
            sentence = []
            # # preprocess["step"] = "spell correction"
            # sentences = sent_tokenize(raw_text)
            # for sent in sentences:
            #     for _, pos in pos_tag(word_tokenize(sent)):
            #         if pos == "NNP":
            #             sentence.append(sent)
            #             break

            for word, pos in pos_tag(word_tokenize(raw_text)):
                if pos == "NNP":
                    sentence.append(word)

            raw_text = " ".join(sentence)

            # preprocess["processed_text"] = raw_text
            # preprocessed.append(preprocess)
            # preprocess = {}

    # preprocess['step'] = "final_text"
    # preprocess["processed_text"] = raw_text
    # preprocessed.append(preprocess)

    return raw_text


def demo_inference(processed_text, input, model_name):
    """

    Parameters
    ----------
    text
    input

    Returns
    -------
    predictions of NER models

    """
    prediction = []
    for model in input:
        predict = {}
        if model["model_name"] == "spacy medium":
            model_path = model["model_path"]
            model_name = predict["model_name"] = "spacy_med"
            predict["model_prediction"] = spacy_prediction(
                processed_text, model_path, model_name
            )
            prediction.append(predict)
        if model["model_name"] == "spacy small":
            model_path = model["model_path"]
            model_name = predict["model_name"] = "spacy_small"
            predict["model_prediction"] = spacy_prediction(
                processed_text, model_path, model_name
            )
            prediction.append(predict)

        if model["model_name"] == "bert":
            model_path = model["model_path"]
            predict["model_name"] = "bert"
            predict["model_prediction"] = bert_prediction(processed_text, model_path)
            prediction.append(predict)

        if model["model_name"] == "crf":
            model_path = model["model_path"]
            predict["model_name"] = "crf"
            predict["model_prediction"] = crf_prediction(processed_text, model_path)
            prediction.append(predict)

        if model["model_name"] == "crf_lstm":
            model_path = model["model_path"]
            predict["model_name"] = "crf_lstm"
            predict["model_prediction"] = crf_lstm_prediction(
                processed_text, model_path
            )
            prediction.append(predict)

        if model["model_name"] == "diet":
            model_path = model["model_path"]
            predict["model_name"] = "diet"
            predict["model_prediction"] = diet_prediction(processed_text, model_path)
            prediction.append(predict)

    return prediction


def inference(text, model_path, model_name="spacy_blank"):
    # inference method would be called inside Flask predict
    if model_name == "spacy_small":
        return spacy_prediction(text, model_path, model_name)
    if model_name == "spacy_med":
        return spacy_prediction(text, model_path, model_name)
    if model_name == "bert":
        return bert_prediction(text, model_path)
    if model_name == "crf":
        return crf_prediction(text, model_path, model_name)
    if model_name == "crf_lstm":
        return crf_lstm_prediction(text, model_path)
    if model_name == "diet":
        return diet_prediction(text, model_path)


def inference_all(json, model_path, model_name):
    data = json.loads(json)
    for doc in data["Attachments"]:
        for page in doc["images"]:
            if page["image_text_process"]:
                processed_text = page["image_text_process"]
                page["image_ner"] = inference(processed_text, model_path, model_name)
            else:
                raw_text = page["image_text_raw"]
                processed_text = text_processing(raw_text)
                page["image_ner"] = inference(processed_text, model_path, model_name)

    return data


text1 = (
    "travellers Casualty and Surety Company only applicable in Guam Puerto Rico and the Virgin Islands IMPORTANT "
    "INSTRUCTIONS.This Application will only be accepted for Privately held commercial companies and Non Profit"
    " organizations with 250 or fewer employees and 100 million or less in assets and 100 million or less in "
    "revenues This Application will not be accepted for Public Companies Government Entities or Financial "
    "Institutions .Applicant means all corporations organizations or other entities including subsidiaries and "
    "Employee Benefit Plans subject to erika that are proposed for this insurance in Item . APPLICANT INFORMATION ."
    " APPLICANT information io Subacute Rehabilitation Center plc a California limited liability company dba 1 . "
    "Name of Applicant Golden Legacy Gare Center Street Address 12260 foothills bled City Sylmarr State CA ZIP "
    "Code 91342 Year Applicant s business was established 2019 2 .the Applicant wish to include additional "
    "entities e.g . affiliates partnerships joint ventures as insured for coverage f Yes attach a list and a "
    "description of each entity . Yes No K Total number of employees at all locations Current Year 300 . Prior"
    " Year 300 Total number of volunteers only if Applicant is a non profit organization total niimbher nf "
    "Incatinne"
)
if __name__ == "__main__":
    test_sentence = """Mr. Trump’s tweets began just moments after a Fox News report by Mike Tobin, a reporter for the
    network, about protests in Minnesota and elsewhere."""
    steps = ["regex", "spell_correction"]
    path = "C:\\Users\\Divakar.Sharma\\Desktop\\Travelers\\AI-Underwriting\\src\\ner\\models\\"
    results = inference(text_processing(test_sentence), path, "diet")
    # input = [{"model_name": "spacy", "model_path": path}, {"model_name": "bert", "model_path": path},
    # input = [{"model_name": "crf_lstm", "model_path": " "}]
    # results = demo_inference(test_sentence, input)
    # results = demo_preprocessing(test_sentence, steps)
    print(results)
